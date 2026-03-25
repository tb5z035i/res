"""RES -- Referring Expression Segmentation."""

from res.api import segment
from res.types import SegmentationResult
from res.backends import get_backend, register_backend, BACKEND_REGISTRY

__all__ = [
    "segment",
    "SegmentationResult",
    "get_backend",
    "register_backend",
    "BACKEND_REGISTRY",
]
