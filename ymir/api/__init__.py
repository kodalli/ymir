from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from .shared import templates
from .functions import router as functions_router
from .generation import router as generation_router
from .conversion import router as conversion_router
from .annotation import router as annotation_router
from .export import router as export_router
from .datasets import router as datasets_router
from ymir.data import get_database

routers = [
    functions_router,
    generation_router,
    conversion_router,
    annotation_router,
    export_router,
    datasets_router,
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup - initialize SQLite database
    db = get_database()
    await db.initialize()
    logger.info("Database initialized")

    yield

    # Shutdown - close database connection
    await db.close()
    logger.info("Database connection closed")


__all__ = ["routers", "templates", "lifespan"]
