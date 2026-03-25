from __future__ import annotations

import os

import numpy as np

from res.backends.base import Backend
from res.types import SegmentationResult

RES_SAM3_CHECKPOINT = "RES_SAM3_CHECKPOINT"


class SAM3Backend(Backend):
    """SAM 3 (Segment Anything with Concepts) backend.

    Requires the ``sam3`` package installed from ``third_party/sam3`` and a
    CUDA-capable GPU with PyTorch >= 2.7.

    Checkpoint resolution order:

    1. Explicit *checkpoint_path* constructor argument.
    2. ``RES_SAM3_CHECKPOINT`` environment variable.
    3. Auto-download from HuggingFace (requires authentication).

    Parameters
    ----------
    device:
        Torch device string.  Defaults to ``"cuda"`` if available.
    confidence_threshold:
        Minimum score to keep a detection (passed to Sam3Processor).
    checkpoint_path:
        Path to a local checkpoint file.  Falls back to the
        ``RES_SAM3_CHECKPOINT`` env var, then to HuggingFace download.
    """

    def __init__(
        self,
        device: str | None = None,
        confidence_threshold: float = 0.5,
        checkpoint_path: str | None = None,
    ) -> None:
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._checkpoint_path = checkpoint_path
        self._model = None
        self._processor = None

    def _resolve_checkpoint(self) -> str | None:
        """Return the checkpoint path, consulting the env var as fallback."""
        return self._checkpoint_path or os.environ.get(RES_SAM3_CHECKPOINT)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = self._resolve_checkpoint()
        self._model = build_sam3_image_model(
            device=device,
            checkpoint_path=ckpt,
            load_from_HF=ckpt is None,
        )
        self._processor = Sam3Processor(
            self._model,
            device=device,
            confidence_threshold=self._confidence_threshold,
        )

    def segment(self, image: np.ndarray, prompt: str) -> list[SegmentationResult]:
        self._ensure_model()
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image)
        state = self._processor.set_image(pil_image)
        state = self._processor.set_text_prompt(prompt, state)

        masks = state["masks"]       # (N, 1, H, W) bool tensor
        boxes = state["boxes"]       # (N, 4) in [x0, y0, x1, y1] pixel coords
        scores = state["scores"]     # (N,)

        results: list[SegmentationResult] = []
        for i in range(len(masks)):
            mask_np = masks[i, 0].cpu().numpy().astype(np.uint8) * 255
            box = boxes[i].cpu().numpy()
            bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            score = float(scores[i].cpu())

            results.append(
                SegmentationResult(
                    mask=mask_np,
                    score=score,
                    label=prompt,
                    bbox=bbox,
                )
            )

        return results

    def is_available(self) -> bool:
        import importlib.util

        return all(
            importlib.util.find_spec(pkg) is not None
            for pkg in ("sam3", "torch", "torchvision")
        )
