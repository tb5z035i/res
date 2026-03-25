from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from res.backends import get_backend
from res.types import SegmentationResult


def segment(
    image: np.ndarray | str | Path,
    prompt: str,
    backend: str = "mock",
    **backend_kwargs: Any,
) -> list[SegmentationResult]:
    """Run referring-expression segmentation.

    Parameters
    ----------
    image:
        RGB image as an ndarray, or a file path (str / Path) to load.
    prompt:
        Natural-language referring expression describing the target.
    backend:
        Name of the registered backend to use.
    **backend_kwargs:
        Extra keyword arguments forwarded to the backend constructor.

    Returns
    -------
    list[SegmentationResult]
    """
    if isinstance(image, (str, Path)):
        image = np.array(Image.open(image).convert("RGB"))

    engine = get_backend(backend, **backend_kwargs)
    return engine.segment(image, prompt)
