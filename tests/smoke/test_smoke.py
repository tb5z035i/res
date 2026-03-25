"""End-to-end smoke tests: import, mock round-trip, CLI subprocess."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


class TestImportAndMockRoundTrip:
    def test_import_res(self) -> None:
        import res
        assert hasattr(res, "segment")
        assert hasattr(res, "SegmentationResult")
        assert "mock" in res.BACKEND_REGISTRY

    def test_mock_segment_roundtrip(self, sample_rgb: np.ndarray) -> None:
        import res

        results = res.segment(sample_rgb, "smoke test", backend="mock")
        assert len(results) >= 1
        r = results[0]
        assert r.mask.shape == sample_rgb.shape[:2]
        assert r.mask.dtype == np.uint8
        assert 0 <= r.score <= 1
        assert r.label == "smoke test"

    def test_segment_from_file(self, sample_image_path: Path) -> None:
        import res

        results = res.segment(sample_image_path, "from file", backend="mock")
        assert len(results) >= 1


class TestCLISubprocess:
    def test_cli_segment_creates_output(self, tmp_path: Path) -> None:
        img_path = tmp_path / "input.png"
        Image.fromarray(np.full((32, 32, 3), 100, dtype=np.uint8)).save(img_path)

        out_dir = tmp_path / "output"
        result = subprocess.run(
            [sys.executable, "-m", "res.cli", "segment",
             "--image", str(img_path),
             "--prompt", "smoke",
             "--output", str(out_dir)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
        assert (out_dir / "mask_0.png").exists()
        assert (out_dir / "overlay_0.png").exists()

    def test_cli_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "res.cli", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "segment" in result.stdout
