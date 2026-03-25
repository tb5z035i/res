from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def sample_rgb() -> np.ndarray:
    """64x64 solid-red RGB image as uint8 ndarray."""
    return np.full((64, 64, 3), fill_value=[255, 0, 0], dtype=np.uint8)


@pytest.fixture()
def sample_image_path(tmp_path: Path, sample_rgb: np.ndarray) -> Path:
    """Write sample_rgb to a temporary PNG file and return the path."""
    p = tmp_path / "sample.png"
    Image.fromarray(sample_rgb).save(p)
    return p
