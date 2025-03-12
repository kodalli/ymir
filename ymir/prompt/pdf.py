from typing import Optional, List, Tuple, Dict
from pypdf import PdfReader, PdfWriter
from tqdm import tqdm
from loguru import logger
from multiprocessing import Pool, cpu_count


def extract_chapter_starts(
    pdf_path: str, toc_page_range: Optional[Tuple[int, int]] = None
) -> List[int]:
    """Extract the starting page numbers for each chapter in the PDF.

    Args:
        pdf_path: Path to the PDF file to analyze
        toc_page_range: Optional tuple (start_page, end_page) with 1-indexed page numbers specifying
                        which pages contain the table of contents. Both start_page and end_page are inclusive.
                        If provided, automatic TOC detection is skipped.

    Returns:
        List[int]: A sorted list of 0-indexed page numbers where chapters start
    """
    reader = PdfReader(pdf_path)
    chapter_starts = []

    logger.info(f"PDF has {len(reader.pages)} pages")

    # First, try to identify Table of Contents pages
    toc_pages = []

    # If toc_page_range is provided, use it directly
    if toc_page_range is not None:
        start_page, end_page = toc_page_range
        # Convert to 0-indexed if needed (PDF users typically use 1-indexed page numbers)
        start_page = max(0, start_page - 1)  # Ensure we don't go below 0
        end_page = min(
            len(reader.pages) - 1, end_page - 1
        )  # Ensure we don't exceed PDF length

        toc_pages = list(range(start_page, end_page + 1))
        logger.info(
            f"Using user-provided TOC page range: pages {start_page + 1}-{end_page + 1}"
        )
    else:
        # Otherwise use automatic detection
        for page_num, page in tqdm(
            enumerate(reader.pages),
            total=len(reader.pages),
            desc="Identifying Table of Contents pages",
        ):
            try:
                text = page.extract_text().strip()
                if text.startswith("Table of Contents"):
                    toc_pages.append(page_num)
                    logger.info(f"Found Table of Contents on page {page_num + 1}")
                    # Also check the next few pages as TOC might span multiple pages
                    for i in range(1, 10):  # Check up to 10 pages after TOC start
                        if page_num + i < len(reader.pages):
                            toc_pages.append(page_num + i)
            except Exception as e:
                logger.warning(
                    f"Error extracting text from page {page_num + 1}: {str(e)}"
                )

        # Special handling: if no TOC found, check the first 10 pages for any page with lots of links
        if not toc_pages:
            logger.info(
                "No Table of Contents page found by text, checking for pages with many links"
            )
            for page_num in tqdm(
                range(min(10, len(reader.pages))),
                desc="Checking for pages with many links",
            ):
                page = reader.pages[page_num]
                if (
                    "/Annots" in page and len(page["/Annots"]) > 0
                ):  # Arbitrary threshold - pages with many links
                    toc_pages.append(page_num)
                    logger.info(
                        f"Page {page_num + 1} has {len(page['/Annots'])} annotations, might be a TOC"
                    )

        # If still couldn't find TOC pages, check all pages
        if not toc_pages:
            logger.warning(
                "Could not identify Table of Contents pages. Checking all pages for links."
            )
            pages_to_check = range(len(reader.pages))
            for page_num in tqdm(
                pages_to_check,
                desc="Checking for pages with many links",
            ):
                page = reader.pages[page_num]
                if (
                    "/Annots" in page and len(page["/Annots"]) > 0
                ):  # Arbitrary threshold - pages with many links
                    toc_pages.append(page_num)
                    logger.info(
                        f"Page {page_num + 1} has {len(page['/Annots'])} annotations, might be a TOC"
                    )

    logger.info(f"Checking {len(toc_pages)} pages for TOC links")

    # Try to extract Links from each PDF object to see their structure
    link_types = {}
    dest_types = {}

    # First pass - analyze link structure in the document
    for page_num in tqdm(
        toc_pages, total=len(toc_pages), desc="Analyzing link structure"
    ):
        page = reader.pages[page_num]
        if "/Annots" in page:
            for annot in page["/Annots"]:
                try:
                    annot_obj = annot.get_object()
                    annot_type = annot_obj.get("/Subtype")

                    # Count types of annotations
                    if annot_type not in link_types:
                        link_types[annot_type] = 0
                    link_types[annot_type] += 1

                    if annot_type == "/Link":
                        # Track what kind of destinations we have
                        if "/Dest" in annot_obj:
                            dest = annot_obj["/Dest"]
                            dest_type = f"Dest:{type(dest).__name__}"
                            if dest_type not in dest_types:
                                dest_types[dest_type] = 0
                            dest_types[dest_type] += 1

                            # If it's a list, look at its structure
                            if isinstance(dest, list) and len(dest) > 0:
                                list_type = (
                                    f"List[{len(dest)}]:{type(dest[0]).__name__}"
                                )
                                if list_type not in dest_types:
                                    dest_types[list_type] = 0
                                dest_types[list_type] += 1

                        elif "/A" in annot_obj:
                            action = annot_obj["/A"]
                            if action.get("/S") == "/GoTo":
                                dest = action.get("/D")
                                dest_type = f"Action:{type(dest).__name__}"
                                if dest_type not in dest_types:
                                    dest_types[dest_type] = 0
                                dest_types[dest_type] += 1

                                # If it's a list, look at its structure
                                if isinstance(dest, list) and len(dest) > 0:
                                    list_type = f"ActionList[{len(dest)}]:{type(dest[0]).__name__}"
                                    if list_type not in dest_types:
                                        dest_types[list_type] = 0
                                    dest_types[list_type] += 1
                except Exception as e:
                    logger.warning(
                        f"Error analyzing annotation on page {page_num + 1}: {str(e)}"
                    )

    logger.info(f"Annotation types found: {link_types}")
    logger.info(f"Destination types found: {dest_types}")

    # Now extract chapter links based on what we learned about the structure
    for page_num in tqdm(
        toc_pages, total=len(toc_pages), desc="Extracting chapter links"
    ):
        page = reader.pages[page_num]
        if "/Annots" in page:
            for annot in page["/Annots"]:
                try:
                    annot_obj = annot.get_object()
                    if annot_obj.get("/Subtype") == "/Link":
                        dest = None
                        if "/Dest" in annot_obj:
                            dest = annot_obj["/Dest"]
                        elif "/A" in annot_obj and annot_obj["/A"].get("/S") == "/GoTo":
                            dest = annot_obj["/A"].get("/D")

                        if dest:
                            if isinstance(dest, str):  # Named destination
                                logger.debug(f"Found named destination: {dest}")
                                dests = reader.root_object.get("/Names", {}).get(
                                    "/Dests", {}
                                )
                                if "/Kids" in dests:
                                    for kid in dests["/Kids"]:
                                        names = kid.get("/Names", [])
                                        for i in range(0, len(names), 2):
                                            if names[i] == dest:
                                                dest = names[i + 1].get("/D")[0]
                                                break

                            # Different methods to resolve the destination
                            page_index = None

                            # Method 1: Direct reference comparison
                            if isinstance(dest, list) and len(dest) > 0:
                                page_ref = dest[0]
                                logger.debug(f"Resolving destination: {page_ref}")

                                # Try different methods to find the page
                                # 1. Compare by indirect reference
                                for i, p in enumerate(reader.pages):
                                    if p.indirect_reference == page_ref:
                                        page_index = i
                                        logger.debug(
                                            f"Found page by indirect reference: {i + 1}"
                                        )
                                        break

                                # 2. For PDFs where the first element is an integer page number
                                if page_index is None and isinstance(
                                    page_ref, (int, float)
                                ):
                                    try:
                                        page_num = int(page_ref)
                                        if 0 <= page_num < len(reader.pages):
                                            page_index = page_num
                                            logger.debug(
                                                f"Found page by direct page number: {page_num + 1}"
                                            )
                                    except (ValueError, TypeError):
                                        pass

                                # 3. Try to get object number from reference
                                if page_index is None and hasattr(page_ref, "idnum"):
                                    ref_id = page_ref.idnum
                                    for i, p in enumerate(reader.pages):
                                        if (
                                            hasattr(p.indirect_reference, "idnum")
                                            and p.indirect_reference.idnum == ref_id
                                        ):
                                            page_index = i
                                            logger.debug(
                                                f"Found page by idnum: {i + 1}"
                                            )
                                            break

                                # 4. Try to extract from the reference string
                                if page_index is None and hasattr(page_ref, "__str__"):
                                    ref_str = str(page_ref)
                                    for i, p in enumerate(reader.pages):
                                        if (
                                            hasattr(p.indirect_reference, "__str__")
                                            and str(p.indirect_reference) == ref_str
                                        ):
                                            page_index = i
                                            logger.debug(
                                                f"Found page by string representation: {i + 1}"
                                            )
                                            break

                            # Method 2: If dest contains page number information
                            if (
                                page_index is None
                                and isinstance(dest, list)
                                and len(dest) > 2
                            ):
                                try:
                                    # Sometimes the 3rd element is the page number
                                    page_num = int(dest[2])
                                    if 0 <= page_num < len(reader.pages):
                                        page_index = page_num
                                        logger.debug(
                                            f"Found page by destination array index 2: {page_num + 1}"
                                        )
                                except (ValueError, TypeError):
                                    pass

                            # If we found a valid page index, add it to chapter starts
                            if (
                                page_index is not None
                                and page_index not in chapter_starts
                            ):
                                chapter_starts.append(page_index)
                            elif isinstance(dest, list):
                                # If we couldn't resolve it, log details for debugging
                                logger.debug(
                                    f"Could not resolve page for destination: {dest}"
                                )
                                if len(dest) > 0:
                                    logger.debug(
                                        f"First element type: {type(dest[0]).__name__}"
                                    )
                                    if hasattr(dest[0], "idnum"):
                                        logger.debug(f"idnum: {dest[0].idnum}")
                                    if hasattr(dest[0], "__str__"):
                                        logger.debug(f"str: {str(dest[0])}")
                except Exception as e:
                    logger.warning(
                        f"Error processing annotation on page {page_num + 1}: {str(e)}"
                    )

    logger.info(
        f"Found {len(chapter_starts)} potential chapter starts at pages: {[p + 1 for p in chapter_starts]}"
    )
    return sorted(chapter_starts)


def process_chapter(
    args: Tuple[str, str, int, List[int], int],
) -> Tuple[int, int, int, str, List[str]]:
    """Process a single chapter extraction - designed for parallel processing"""
    pdf_path, output_prefix, i, chapter_starts, num_chapters = args
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    start_page = chapter_starts[i]
    end_page = chapter_starts[i + 1]

    chapter_title = f"Chapter_{i + 1}"

    # Add all pages in this chapter to the output PDF
    pages_content = []
    for j in range(start_page, end_page):
        writer.add_page(reader.pages[j])
        pages_content.append(reader.pages[j].extract_text())

    # Use a more readable filename format
    output_path = f"{output_prefix}_{chapter_title}.pdf"
    with open(output_path, "wb") as output_file:
        writer.write(output_file)

    return i, start_page, end_page, output_path, pages_content


def split_pdf_by_chapters(
    pdf_path: str,
    output_prefix: str,
    chapter_starts: Optional[List[int]] = None,
    num_workers: Optional[int] = None,
) -> Dict[str, List[str]]:
    reader = PdfReader(pdf_path)

    if chapter_starts is None:
        chapter_starts = extract_chapter_starts(pdf_path)

    if not chapter_starts:
        logger.info(
            "No chapter start pages found. Check if the PDF has a Table of Contents with links."
        )
        return

    logger.info(
        f"Found {len(chapter_starts)} potential chapter start points at pages: {chapter_starts}"
    )

    chapter_starts.append(len(reader.pages))  # Add end of last chapter
    num_chapters = len(chapter_starts) - 1

    # Set the number of workers for the pool
    if num_workers is None:
        num_workers = min(
            cpu_count(), num_chapters
        )  # Don't use more workers than chapters

    logger.info(
        f"Processing {num_chapters} chapters using {num_workers} worker threads"
    )

    # Create tasks for the pool
    tasks = [
        (pdf_path, output_prefix, i, chapter_starts, num_chapters)
        for i in range(num_chapters)
    ]

    # Process chapters in parallel
    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_chapter, tasks),
                total=num_chapters,
                desc="Splitting chapters",
            )
        )

    # Log results
    final_pdf_paths = {}
    for i, start_page, end_page, output_path, pages_content in sorted(results):
        logger.info(
            f"Chapter {i + 1}: (Pages {start_page + 1}-{end_page}) saved as {output_path}"
        )
        final_pdf_paths[output_path] = pages_content

    return final_pdf_paths
