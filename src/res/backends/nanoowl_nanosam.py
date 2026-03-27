from __future__ import annotations

import os

import numpy as np

from res.backends.base import Backend
from res.types import SegmentationResult

RES_NANOOWL_MODEL_NAME = "RES_NANOOWL_MODEL_NAME"
RES_NANOOWL_ENGINE = "RES_NANOOWL_ENGINE"
RES_NANOSAM_IMAGE_ENCODER_ENGINE = "RES_NANOSAM_IMAGE_ENCODER_ENGINE"
RES_NANOSAM_MASK_DECODER_ENGINE = "RES_NANOSAM_MASK_DECODER_ENGINE"


class NanoOwlNanoSamBackend(Backend):
    """NanoOWL (open-set detection) + NanoSAM (TRT segmentation) backend.

    Requires three pre-built TensorRT engine files:

    * OWL-ViT image encoder  (via NanoOWL)
    * NanoSAM image encoder   (ResNet-18 distilled)
    * NanoSAM mask decoder    (MobileSAM decoder)

    All paths are resolved in order:

    1. Explicit constructor arguments.
    2. Environment variables (``RES_NANOOWL_MODEL_NAME``,
       ``RES_NANOOWL_ENGINE``, ``RES_NANOSAM_IMAGE_ENCODER_ENGINE``,
       ``RES_NANOSAM_MASK_DECODER_ENGINE``).

    Parameters
    ----------
    owl_model_name:
        Local path to OWL-ViT weights directory, or HuggingFace model id.
    owl_engine_path:
        Path to the NanoOWL image-encoder TRT engine.
    sam_image_encoder_engine_path:
        Path to the NanoSAM image-encoder TRT engine.
    sam_mask_decoder_engine_path:
        Path to the NanoSAM mask-decoder TRT engine.
    detection_threshold:
        Minimum OWL-ViT detection score.
    device:
        Torch device string.
    """

    def __init__(
        self,
        owl_model_name: str | None = None,
        owl_engine_path: str | None = None,
        sam_image_encoder_engine_path: str | None = None,
        sam_mask_decoder_engine_path: str | None = None,
        detection_threshold: float = 0.1,
        device: str = "cuda",
    ) -> None:
        self._owl_model_name = (
            owl_model_name
            or os.environ.get(RES_NANOOWL_MODEL_NAME)
            or "google/owlvit-base-patch32"
        )
        self._owl_engine_path = owl_engine_path
        self._sam_image_encoder_engine_path = sam_image_encoder_engine_path
        self._sam_mask_decoder_engine_path = sam_mask_decoder_engine_path
        self._detection_threshold = detection_threshold
        self._device = device

        self._owl_predictor = None
        self._sam_predictor = None

    def _resolve_engine(self, explicit: str | None, env_var: str, label: str) -> str:
        path = explicit or os.environ.get(env_var)
        if path is None:
            raise RuntimeError(
                f"No TRT engine path for {label}. "
                f"Pass it to the constructor or set the {env_var} environment variable."
            )
        return path

    def _ensure_models(self) -> None:
        if self._owl_predictor is not None:
            return

        from nanoowl.owl_predictor import OwlPredictor
        from nanosam.utils.predictor import Predictor as NanoSAMPredictor

        owl_engine = self._resolve_engine(
            self._owl_engine_path, RES_NANOOWL_ENGINE, "NanoOWL image encoder"
        )
        sam_enc = self._resolve_engine(
            self._sam_image_encoder_engine_path,
            RES_NANOSAM_IMAGE_ENCODER_ENGINE,
            "NanoSAM image encoder",
        )
        sam_dec = self._resolve_engine(
            self._sam_mask_decoder_engine_path,
            RES_NANOSAM_MASK_DECODER_ENGINE,
            "NanoSAM mask decoder",
        )

        self._owl_predictor = OwlPredictor(
            model_name=self._owl_model_name,
            device=self._device,
            image_encoder_engine=owl_engine,
        )
        self._sam_predictor = NanoSAMPredictor(sam_enc, sam_dec)

    def segment(self, image: np.ndarray, prompt: str) -> list[SegmentationResult]:
        self._ensure_models()
        from PIL import Image as PILImage
        import time as _time, json as _json

        pil_image = PILImage.fromarray(image)

        # #region agent log
        _t0 = _time.perf_counter()
        # #endregion

        if not hasattr(self, '_text_cache') or self._text_cache_key != prompt:
            self._text_cache = self._owl_predictor.encode_text([prompt])
            self._text_cache_key = prompt

        # #region agent log
        _t1 = _time.perf_counter()
        # #endregion

        detections = self._owl_predictor.predict(
            image=pil_image,
            text=[prompt],
            text_encodings=self._text_cache,
            threshold=self._detection_threshold,
        )

        # #region agent log
        _t2 = _time.perf_counter()
        # #endregion

        if len(detections.boxes) == 0:
            return []

        # #region agent log
        _t3 = _time.perf_counter()
        # #endregion

        self._sam_predictor.set_image(pil_image)

        # #region agent log
        _t4 = _time.perf_counter()
        # #endregion

        results: list[SegmentationResult] = []
        for i in range(len(detections.boxes)):
            box = detections.boxes[i].detach().cpu().numpy()
            score = float(detections.scores[i].detach().cpu())

            x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            points = np.array([[x0, y0], [x1, y1]])
            point_labels = np.array([2, 3])

            mask_tensor, iou_preds, _ = self._sam_predictor.predict(
                points, point_labels
            )
            best_idx = iou_preds.argmax().item()
            mask_bool = mask_tensor[0, best_idx].detach().cpu().numpy() > 0
            mask_np = mask_bool.astype(np.uint8) * 255

            results.append(
                SegmentationResult(
                    mask=mask_np,
                    score=score,
                    label=prompt,
                    bbox=(x0, y0, x1, y1),
                )
            )

        # #region agent log
        _t5 = _time.perf_counter()
        _log = {"sessionId":"2b5219","location":"nanoowl_nanosam.py:segment","message":"timing","data":{"text_encode_ms":(_t1-_t0)*1000,"owl_predict_ms":(_t2-_t1)*1000,"sam_set_image_ms":(_t4-_t3)*1000,"sam_decode_loop_ms":(_t5-_t4)*1000,"total_ms":(_t5-_t0)*1000,"num_detections":len(detections.boxes)},"timestamp":_t5}
        with open("/home/nvidia/workspace/res/.cursor/debug-2b5219.log","a") as _f: _f.write(_json.dumps(_log)+"\n")
        # #endregion

        return results

    def is_available(self) -> bool:
        import importlib.util

        return all(
            importlib.util.find_spec(pkg) is not None
            for pkg in ("nanoowl", "nanosam", "tensorrt", "torch2trt")
        )
