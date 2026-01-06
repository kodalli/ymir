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
from .actors import router as actors_router
from .scenarios import router as scenarios_router
from .templates_api import router as templates_router
from .actor_templates import router as actor_templates_router
from ymir.data import get_database

routers = [
    functions_router,
    generation_router,
    conversion_router,
    annotation_router,
    export_router,
    datasets_router,
    actors_router,
    scenarios_router,
    templates_router,
    actor_templates_router,
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
