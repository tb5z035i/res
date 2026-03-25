from __future__ import annotations

import numpy as np

from res.backends.mock import MockBackend


class TestMockBackend:
    def test_returns_one_result(self, sample_rgb: np.ndarray) -> None:
        results = MockBackend().segment(sample_rgb, "anything")
        assert len(results) == 1

    def test_mask_shape_matches_input(self, sample_rgb: np.ndarray) -> None:
        r = MockBackend().segment(sample_rgb, "test")[0]
        h, w = sample_rgb.shape[:2]
        assert r.mask.shape == (h, w)

    def test_mask_dtype_uint8(self, sample_rgb: np.ndarray) -> None:
        r = MockBackend().segment(sample_rgb, "test")[0]
        assert r.mask.dtype == np.uint8

    def test_mask_binary_values(self, sample_rgb: np.ndarray) -> None:
        r = MockBackend().segment(sample_rgb, "test")[0]
        assert set(np.unique(r.mask)) <= {0, 255}

    def test_score_is_one(self, sample_rgb: np.ndarray) -> None:
        r = MockBackend().segment(sample_rgb, "test")[0]
        assert r.score == 1.0

    def test_label_echoes_prompt(self, sample_rgb: np.ndarray) -> None:
        r = MockBackend().segment(sample_rgb, "red car")[0]
        assert r.label == "red car"

    def test_bbox_within_image(self, sample_rgb: np.ndarray) -> None:
        r = MockBackend().segment(sample_rgb, "test")[0]
        x1, y1, x2, y2 = r.bbox
        h, w = sample_rgb.shape[:2]
        assert 0 <= x1 <= x2 < w
        assert 0 <= y1 <= y2 < h

    def test_deterministic(self, sample_rgb: np.ndarray) -> None:
        a = MockBackend().segment(sample_rgb, "x")[0]
        b = MockBackend().segment(sample_rgb, "x")[0]
        np.testing.assert_array_equal(a.mask, b.mask)
        assert a.bbox == b.bbox

    def test_tiny_image(self) -> None:
        tiny = np.zeros((1, 1, 3), dtype=np.uint8)
        results = MockBackend().segment(tiny, "dot")
        assert len(results) == 1

    def test_empty_prompt(self, sample_rgb: np.ndarray) -> None:
        r = MockBackend().segment(sample_rgb, "")[0]
        assert r.label == ""

    def test_is_available(self) -> None:
        assert MockBackend().is_available() is True
