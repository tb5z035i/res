from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import res
from res.types import SegmentationResult


class TestSegmentFunction:
    def test_with_ndarray(self, sample_rgb: np.ndarray) -> None:
        results = res.segment(sample_rgb, "hello", backend="mock")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert isinstance(results[0], SegmentationResult)

    def test_with_file_path_string(self, sample_image_path: Path) -> None:
        results = res.segment(str(sample_image_path), "hello", backend="mock")
        assert len(results) >= 1

    def test_with_file_path_object(self, sample_image_path: Path) -> None:
        results = res.segment(sample_image_path, "hello", backend="mock")
        assert len(results) >= 1

    def test_unknown_backend_raises(self, sample_rgb: np.ndarray) -> None:
        with pytest.raises(KeyError, match="Unknown backend"):
            res.segment(sample_rgb, "hello", backend="does_not_exist")

    def test_dispatches_to_correct_backend(self, sample_rgb: np.ndarray) -> None:
        results = res.segment(sample_rgb, "prompt", backend="mock")
        assert results[0].label == "prompt"
        assert results[0].score == 1.0
