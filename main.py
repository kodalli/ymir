#!/usr/bin/env python3
"""
Main entrypoint for Ymir Dataset Builder
"""

import argparse
import os
import uvicorn
from loguru import logger


def setup_directories():
    """Set up necessary directories for the application"""
    dirs = ["ymir/templates", "ymir/static", "ymir/static/css", "ymir/static/js"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Directories checked/created")


def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(
        description="Ymir Dataset Builder - FastAPI Backend"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8008, help="Port to run the server on"
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    # Set up directories if they don't exist
    setup_directories()

    # Start FastAPI server
    logger.info(f"Starting Ymir Dataset Builder on http://{args.host}:{args.port}")
    uvicorn.run(
        "ymir.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
