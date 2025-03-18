from .document_processing import router as document_processing_router
from .batch_processing import router as batch_processing_router
from .datasets_processing import router as datasets_processing_router
from .rlhf_processing import router as rlhf_processing_router
from .rlhf_processing import lifespan
from .triplet_processing import router as triplet_processing_router
from .shared import router as shared_router
from .shared import templates

routers = [
    document_processing_router,
    batch_processing_router,
    datasets_processing_router,
    rlhf_processing_router,
    triplet_processing_router,
    shared_router,
]

__all__ = ["routers", "templates", "lifespan"]
