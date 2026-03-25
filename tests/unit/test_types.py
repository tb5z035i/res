from __future__ import annotations

import numpy as np

from res.types import SegmentationResult


class TestSegmentationResult:
    def test_construction(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        r = SegmentationResult(mask=mask, score=0.9, label="cat", bbox=(1, 2, 3, 4))
        assert r.score == 0.9
        assert r.label == "cat"
        assert r.bbox == (1, 2, 3, 4)

    def test_mask_shape_and_dtype(self) -> None:
        mask = np.full((20, 30), 255, dtype=np.uint8)
        r = SegmentationResult(mask=mask, score=1.0, label="dog", bbox=(0, 0, 29, 19))
        assert r.mask.shape == (20, 30)
        assert r.mask.dtype == np.uint8

    def test_mask_binary_values(self) -> None:
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1:4, 1:4] = 255
        r = SegmentationResult(mask=mask, score=0.5, label="box", bbox=(1, 1, 3, 3))
        unique = set(np.unique(r.mask))
        assert unique <= {0, 255}
