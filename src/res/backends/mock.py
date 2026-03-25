from __future__ import annotations

import numpy as np

from res.backends.base import Backend
from res.types import SegmentationResult


class MockBackend(Backend):
    """Deterministic mock backend for testing -- no GPU or model required."""

    def segment(self, image: np.ndarray, prompt: str) -> list[SegmentationResult]:
        h, w = image.shape[:2]

        # Deterministic elliptical mask centred on the image
        cy, cx = h / 2, w / 2
        ry, rx = h / 4, w / 4

        yy, xx = np.ogrid[:h, :w]
        ellipse = ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2
        mask = (ellipse <= 1.0).astype(np.uint8) * 255

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            bbox = (0, 0, 0, 0)
        else:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        return [
            SegmentationResult(
                mask=mask,
                score=1.0,
                label=prompt,
                bbox=bbox,
            )
        ]
