import time
import os
import csv
from pathlib import Path
from typing import Optional
from loguru import logger
from fastapi import Request, Form, UploadFile, File, APIRouter
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from ymir.prompt.pdf import extract_chapter_starts, split_pdf_by_chapters


router = APIRouter()
templates = Jinja2Templates(directory="ymir/templates")

# Progress tracking for PDF processing
pdf_processing_progress = {}


def update_progress(
    job_id: str, step: str, current: int, total: int, message: str = ""
):
    """Update the progress for a given job ID"""
    progress = {
        "step": step,
        "current": current,
        "total": total,
        "percent": int((current / max(total, 1)) * 100),
        "message": message,
        "time": time.time(),
    }
    pdf_processing_progress[job_id] = progress
    logger.info(f"Progress update for job {job_id}: {progress['percent']}% - {message}")
    return progress


@router.get("/pdf_progress/{job_id}")
async def get_pdf_progress(job_id: str):
    """Get the progress for a PDF processing job"""
    if job_id in pdf_processing_progress:
        return pdf_processing_progress[job_id]
    else:
        return {"error": "Job not found", "percent": 0, "message": "Unknown job ID"}


@router.get("/document", response_class=HTMLResponse)
async def document_page(request: Request):
    """Render the Document Processor page"""
    return templates.TemplateResponse(
        "document_processing/document_processor.html",
        {
            "request": request,
        },
    )


@router.post("/upload_pdf")
async def upload_pdf(request: Request, pdf_file: UploadFile = File(...)):
    """Upload a PDF file and return basic information about it"""
    try:
        logger.info(f"PDF upload received: {pdf_file.filename}")

        # Create data directory if it doesn't exist
        data_dir = Path("ymir/data/documents")
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using data directory: {data_dir.absolute()}")

        # Save the uploaded file
        timestamp = int(time.time())
        filename = f"{timestamp}_{pdf_file.filename}"
        file_path = data_dir / filename
        logger.info(f"Saving file to: {file_path.absolute()}")

        with open(file_path, "wb") as f:
            content = await pdf_file.read()
            f.write(content)
            logger.info(f"Wrote {len(content)} bytes to file")

        # Get basic PDF info
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        num_pages = len(reader.pages)
        logger.info(f"PDF has {num_pages} pages")

        # Return PDF info
        response = templates.TemplateResponse(
            "document_processing/document_pdf_info.html",
            {
                "request": request,
                "filename": pdf_file.filename,
                "file_path": str(file_path),
                "num_pages": num_pages,
            },
        )
        logger.info("Returning PDF info template response")
        return response
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}", exc_info=True)
        return templates.TemplateResponse(
            "document_processing/document_error.html",
            {
                "request": request,
                "error_message": f"Error uploading PDF: {str(e)}",
            },
        )


@router.post("/detect_toc")
async def detect_toc(
    request: Request,
    pdf_path: str = Form(...),
    toc_start_page: Optional[int] = Form(None),
    toc_end_page: Optional[int] = Form(None),
):
    """Detect table of contents in a PDF file"""
    try:
        logger.info(f"Detecting TOC in PDF: {pdf_path}")
        logger.info(f"TOC page range provided: {toc_start_page}-{toc_end_page}")

        # Check if manual TOC range was provided
        toc_page_range = None
        if toc_start_page is not None and toc_end_page is not None:
            toc_page_range = (toc_start_page, toc_end_page)
            logger.info(f"Using manually specified TOC page range: {toc_page_range}")

        # Extract chapter starts using the function from pdf.py
        chapter_starts = extract_chapter_starts(pdf_path, toc_page_range)
        logger.info(f"Detected chapter starts (0-indexed): {chapter_starts}")

        if not chapter_starts:
            logger.warning("No chapter starts detected in the PDF")
            return templates.TemplateResponse(
                "document_processing/document_error.html",
                {
                    "request": request,
                    "error_message": "No chapters detected in the PDF. Try specifying a manual TOC page range where the table of contents is located.",
                },
            )

        # Convert to 1-indexed for display
        chapter_pages = [page + 1 for page in chapter_starts]
        logger.info(f"Chapter pages (1-indexed for display): {chapter_pages}")

        # Create chapter info
        chapters = []
        for i, page in enumerate(chapter_pages):
            next_page = (
                chapter_pages[i + 1] - 1 if i < len(chapter_pages) - 1 else "end"
            )
            chapters.append(
                {
                    "number": i + 1,
                    "start_page": page,
                    "end_page": next_page,
                }
            )

        # Return chapter info
        return templates.TemplateResponse(
            "document_processing/document_toc_results.html",
            {
                "request": request,
                "chapters": chapters,
                "chapter_starts": ",".join(map(str, chapter_starts)),
                "pdf_path": pdf_path,
            },
        )
    except Exception as e:
        logger.error(f"Error detecting TOC: {e}", exc_info=True)
        return templates.TemplateResponse(
            "document_processing/document_error.html",
            {
                "request": request,
                "error_message": f"Error detecting table of contents: {str(e)}",
            },
        )


@router.post("/process_pdf")
async def process_pdf(
    request: Request,
    pdf_path: str = Form(...),
    chapter_starts: str = Form(...),
    split_chapters: bool = Form(False),
    extract_text: bool = Form(True),
    create_csv: bool = Form(True),
    toc_start_page: Optional[int] = Form(None),
    toc_end_page: Optional[int] = Form(None),
):
    """Process a PDF file based on detected chapters"""
    try:
        # Generate a unique job ID for tracking progress
        job_id = f"pdf_job_{int(time.time())}"

        # Initialize progress tracking
        update_progress(job_id, "init", 0, 100, "Starting PDF processing")

        # Parse chapter starts
        chapter_starts_list = [int(page) for page in chapter_starts.split(",")]
        update_progress(
            job_id, "chapters", 1, 3, f"Identified {len(chapter_starts_list)} chapters"
        )

        # Create output directory
        timestamp = int(time.time())
        output_dir = Path("ymir/data/documents") / f"output_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        update_progress(job_id, "prepare", 2, 3, "Created output directory")

        # Base filename for outputs
        pdf_filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(pdf_filename)[0]
        output_prefix = str(output_dir / base_name)

        results = {
            "chapters_processed": 0,
            "csv_path": None,
            "chapter_pdfs": [],
            "job_id": job_id,  # Include job ID in results
        }

        update_progress(job_id, "processing", 3, 3, "Starting PDF operations")

        # Split PDF if requested
        if split_chapters:
            # Define a progress callback to track PDF splitting
            def progress_callback(current, total, message=""):
                # progress_percent = 10 + int(
                #     (current / total) * 60
                # )  # Scale to 10-70% range
                update_progress(job_id, "splitting", current, total, message)

            update_progress(
                job_id, "splitting", 0, 100, "Starting PDF chapter splitting"
            )

            # Call split_pdf_by_chapters with the progress callback
            chapter_contents = split_pdf_by_chapters(
                pdf_path,
                output_prefix,
                chapter_starts_list,
                progress_callback=progress_callback,
            )

            results["chapters_processed"] = len(chapter_contents)
            results["chapter_pdfs"] = list(chapter_contents.keys())

            update_progress(
                job_id,
                "splitting",
                100,
                100,
                f"Split PDF into {len(chapter_contents)} chapters",
            )

        # Create CSV if requested
        if create_csv:
            csv_path = f"{output_prefix}_chapters.csv"
            update_progress(job_id, "csv", 0, 100, "Creating CSV dataset")

            # If we already have chapter contents from splitting
            if split_chapters and extract_text:
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["chapter", "pages", "content"])

                    total_chapters = len(chapter_contents)
                    for i, (pdf_path, content) in enumerate(chapter_contents.items()):
                        chapter_num = i + 1
                        start_page = chapter_starts_list[i] + 1  # Convert to 1-indexed
                        end_page = (
                            chapter_starts_list[i + 1]
                            if i + 1 < len(chapter_starts_list)
                            else "end"
                        )
                        pages = f"{start_page}-{end_page}"

                        # Join all page content
                        full_content = "\n\n".join(content.strip())
                        writer.writerow([chapter_num, pages, full_content.strip()])

                        # Update progress (70-90% range for CSV creation)
                        # progress_percent = 70 + int((i / total_chapters) * 20)
                        update_progress(
                            job_id,
                            "csv",
                            i + 1,
                            total_chapters,
                            f"Processing chapter {chapter_num} for CSV",
                        )

            # If we need to extract text without splitting
            elif extract_text:
                update_progress(
                    job_id, "extracting_text", 0, 100, "Extracting text from PDF"
                )
                from pypdf import PdfReader

                reader = PdfReader(pdf_path)

                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["chapter", "pages", "content"])

                    total_chapters = len(chapter_starts_list) - 1
                    for i in range(total_chapters):
                        chapter_num = i + 1
                        start_page = chapter_starts_list[i]
                        end_page = chapter_starts_list[i + 1]
                        pages = f"{start_page + 1}-{end_page}"

                        # Extract text from all pages in this chapter
                        chapter_content = []
                        total_pages = end_page - start_page
                        for page_idx, page_num in enumerate(
                            range(start_page, end_page)
                        ):
                            if page_num < len(reader.pages):
                                page_text = (
                                    reader.pages[page_num].extract_text().strip()
                                )
                                chapter_content.append(page_text)

                                # Update progress for each page processed
                                # sub_progress = int(
                                #     (page_idx / max(total_pages, 1)) * 100
                                update_progress(
                                    job_id,
                                    "extracting_page",
                                    page_idx + 1,
                                    total_pages,
                                    f"Extracting text from chapter {chapter_num}, page {page_num + 1}",
                                )

                        full_content = "\n\n".join(chapter_content)
                        writer.writerow([chapter_num, pages, full_content])

                        # Update overall progress (70-90% range)
                        # progress_percent = 70 + int((i / total_chapters) * 20)
                        update_progress(
                            job_id,
                            "csv",
                            i + 1,
                            total_chapters,
                            f"Added chapter {chapter_num} to CSV",
                        )

            # Just create a CSV with chapter info but no content
            else:
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["chapter", "pages", "content"])

                    total_chapters = len(chapter_starts_list) - 1
                    for i in range(total_chapters):
                        chapter_num = i + 1
                        start_page = chapter_starts_list[i] + 1  # Convert to 1-indexed
                        end_page = chapter_starts_list[i + 1]
                        pages = f"{start_page}-{end_page}"

                        writer.writerow([chapter_num, pages, ""])

                        # Update progress
                        # progress_percent = 70 + int((i / total_chapters) * 20)
                        update_progress(
                            job_id,
                            "csv",
                            i + 1,
                            total_chapters,
                            f"Added chapter {chapter_num} info to CSV",
                        )

            results["csv_path"] = csv_path
            update_progress(job_id, "completed", 100, 100, "PDF processing complete")

        # Return processing results
        return templates.TemplateResponse(
            "document_processing/document_processing_results.html",
            {
                "request": request,
                "results": results,
                "pdf_path": pdf_path,
            },
        )
    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        return templates.TemplateResponse(
            "document_processing/document_error.html",
            {
                "request": request,
                "error_message": f"Error processing PDF: {str(e)}",
            },
        )


if __name__ == "__main__":
    pass
