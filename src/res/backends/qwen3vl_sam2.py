from __future__ import annotations

import numpy as np

from res.backends.base import Backend
from res.types import SegmentationResult


class Qwen3VLSAM2Backend(Backend):
    """Qwen3-VL + SAM2 backend (placeholder -- not yet implemented)."""

    def segment(self, image: np.ndarray, prompt: str) -> list[SegmentationResult]:
        raise NotImplementedError(
            "Qwen3VLSAM2Backend is not yet implemented. "
            "Install the required dependencies and check back later."
        )

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False
