from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from res.cli import main


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def input_image(tmp_path: Path) -> Path:
    img = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
    p = tmp_path / "input.png"
    img.save(p)
    return p


class TestSegmentCommand:
    def test_exit_code_zero(self, runner: CliRunner, input_image: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        result = runner.invoke(main, [
            "segment",
            "--image", str(input_image),
            "--prompt", "test",
            "--output", str(out_dir),
        ])
        assert result.exit_code == 0, result.output

    def test_creates_mask_files(self, runner: CliRunner, input_image: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        runner.invoke(main, [
            "segment",
            "--image", str(input_image),
            "--prompt", "object",
            "--output", str(out_dir),
        ])
        assert (out_dir / "mask_0.png").exists()
        assert (out_dir / "overlay_0.png").exists()

    def test_output_summary(self, runner: CliRunner, input_image: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        result = runner.invoke(main, [
            "segment",
            "--image", str(input_image),
            "--prompt", "thing",
            "--output", str(out_dir),
        ])
        assert "1 result(s)" in result.output
        assert "score=" in result.output

    def test_missing_image_fails(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(main, [
            "segment",
            "--image", "/nonexistent.png",
            "--prompt", "x",
            "--output", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0


class TestUICommand:
    def test_missing_gradio_message(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        """If gradio import fails, the CLI should show a helpful error."""
        import sys
        import importlib

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "res.ui":
                raise ImportError("No module named 'gradio'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        result = runner.invoke(main, ["ui"])
        assert result.exit_code != 0
