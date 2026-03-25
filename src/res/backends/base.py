from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from res.types import SegmentationResult


class Backend(ABC):
    """Abstract base class for segmentation backends."""

    @abstractmethod
    def segment(self, image: np.ndarray, prompt: str) -> list[SegmentationResult]:
        """Run segmentation on *image* given a text *prompt*.

        Parameters
        ----------
        image:
            RGB image as (H, W, 3) uint8 ndarray.
        prompt:
            Natural-language referring expression.

        Returns
        -------
        list[SegmentationResult]
        """

    def is_available(self) -> bool:
        """Return True if all runtime deps for this backend are satisfied."""
        return True
