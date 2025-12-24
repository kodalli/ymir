from contextlib import asynccontextmanager
from fastapi import FastAPI

from .shared import templates
from .functions import router as functions_router
from .generation import router as generation_router
from .conversion import router as conversion_router
from .annotation import router as annotation_router
from .export import router as export_router

routers = [
    functions_router,
    generation_router,
    conversion_router,
    annotation_router,
    export_router,
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    yield
    # Shutdown


__all__ = ["routers", "templates", "lifespan"]
