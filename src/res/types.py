from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class SegmentationResult:
    """A single segmentation result from a backend."""

    mask: np.ndarray
    """(H, W) binary mask, dtype=uint8, values 0 or 255."""

    score: float
    """Confidence score in [0, 1]."""

    label: str
    """Matched label / phrase."""

    bbox: tuple[int, int, int, int]
    """Bounding box as (x1, y1, x2, y2)."""
