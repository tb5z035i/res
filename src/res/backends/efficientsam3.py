from __future__ import annotations

import os

import numpy as np

from res.backends.base import Backend
from res.types import SegmentationResult

RES_EFFICIENTSAM3_CHECKPOINT = "RES_EFFICIENTSAM3_CHECKPOINT"


class EfficientSAM3Backend(Backend):
    """EfficientSAM3 (Stage 1 distilled encoders) backend.

    Uses lightweight image backbones (TinyViT, EfficientViT, RepViT) and
    distilled text encoders (MobileCLIP) from the EfficientSAM3 project,
    running through the SAM 3 processor pipeline.

    Checkpoint resolution order:

    1. Explicit *checkpoint_path* constructor argument.
    2. ``RES_EFFICIENTSAM3_CHECKPOINT`` environment variable.
    3. Auto-download from HuggingFace.

    Parameters
    ----------
    device:
        Torch device string.  Defaults to auto-detect.
    confidence_threshold:
        Minimum score to keep a detection (passed to Sam3Processor).
    checkpoint_path:
        Path to a local checkpoint file.  Falls back to the
        ``RES_EFFICIENTSAM3_CHECKPOINT`` env var, then to HuggingFace.
    backbone_type:
        Image encoder backbone family: ``"tinyvit"``, ``"efficientvit"``,
        or ``"repvit"``.
    model_name:
        Model size within the backbone family.  E.g. ``"5m"``, ``"11m"``,
        ``"21m"`` for TinyViT; ``"b0"``, ``"b1"``, ``"b2"`` for
        EfficientViT; ``"m0_9"``, ``"m1_1"``, ``"m2_3"`` for RepViT.
    text_encoder_type:
        Distilled text encoder variant, e.g. ``"MobileCLIP-S1"``.
        Pass ``None`` to use the full SAM 3 text encoder (requires the
        corresponding checkpoint).
    """

    def __init__(
        self,
        device: str | None = None,
        confidence_threshold: float = 0.5,
        checkpoint_path: str | None = None,
        backbone_type: str = "tinyvit",
        model_name: str = "11m",
        text_encoder_type: str | None = "MobileCLIP-S1",
    ) -> None:
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._checkpoint_path = checkpoint_path
        self._backbone_type = backbone_type
        self._model_name = model_name
        self._text_encoder_type = text_encoder_type
        self._model = None
        self._processor = None

    def _resolve_checkpoint(self) -> str | None:
        """Return the checkpoint path, consulting the env var as fallback."""
        return self._checkpoint_path or os.environ.get(RES_EFFICIENTSAM3_CHECKPOINT)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from sam3.model_builder import build_efficientsam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = self._resolve_checkpoint()
        self._model = build_efficientsam3_image_model(
            device=device,
            checkpoint_path=ckpt,
            load_from_HF=ckpt is None,
            backbone_type=self._backbone_type,
            model_name=self._model_name,
            text_encoder_type=self._text_encoder_type,
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
            for pkg in ("sam3", "torch", "torchvision", "timm")
        )
