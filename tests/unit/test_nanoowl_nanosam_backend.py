from __future__ import annotations

import pytest

from res.backends.nanoowl_nanosam import (
    NanoOwlNanoSamBackend,
    RES_NANOOWL_ENGINE,
    RES_NANOSAM_IMAGE_ENCODER_ENGINE,
    RES_NANOSAM_MASK_DECODER_ENGINE,
)


class TestNanoOwlNanoSamBackend:
    def test_registered(self) -> None:
        from res.backends import BACKEND_REGISTRY

        assert "nanoowl_nanosam" in BACKEND_REGISTRY
        assert BACKEND_REGISTRY["nanoowl_nanosam"] is NanoOwlNanoSamBackend

    def test_is_available_returns_bool(self) -> None:
        result = NanoOwlNanoSamBackend().is_available()
        assert isinstance(result, bool)

    def test_constructor_stores_params(self) -> None:
        b = NanoOwlNanoSamBackend(
            owl_model_name="google/owlvit-base-patch16",
            owl_engine_path="/tmp/owl.engine",
            sam_image_encoder_engine_path="/tmp/sam_enc.engine",
            sam_mask_decoder_engine_path="/tmp/sam_dec.engine",
            detection_threshold=0.25,
            device="cuda:1",
        )
        assert b._owl_model_name == "google/owlvit-base-patch16"
        assert b._owl_engine_path == "/tmp/owl.engine"
        assert b._sam_image_encoder_engine_path == "/tmp/sam_enc.engine"
        assert b._sam_mask_decoder_engine_path == "/tmp/sam_dec.engine"
        assert b._detection_threshold == 0.25
        assert b._device == "cuda:1"
        assert b._owl_predictor is None
        assert b._sam_predictor is None

    def test_owl_model_name_configurable(self) -> None:
        b1 = NanoOwlNanoSamBackend()
        assert b1._owl_model_name == "google/owlvit-base-patch32"

        b2 = NanoOwlNanoSamBackend(owl_model_name="google/owlvit-large-patch14")
        assert b2._owl_model_name == "google/owlvit-large-patch14"

    def test_resolve_engine_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(RES_NANOOWL_ENGINE, "/env/owl.engine")
        monkeypatch.setenv(RES_NANOSAM_IMAGE_ENCODER_ENGINE, "/env/sam_enc.engine")
        monkeypatch.setenv(RES_NANOSAM_MASK_DECODER_ENGINE, "/env/sam_dec.engine")

        b = NanoOwlNanoSamBackend()
        assert b._resolve_engine(None, RES_NANOOWL_ENGINE, "owl") == "/env/owl.engine"
        assert (
            b._resolve_engine(None, RES_NANOSAM_IMAGE_ENCODER_ENGINE, "sam enc")
            == "/env/sam_enc.engine"
        )
        assert (
            b._resolve_engine(None, RES_NANOSAM_MASK_DECODER_ENGINE, "sam dec")
            == "/env/sam_dec.engine"
        )

    def test_resolve_engine_explicit_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(RES_NANOOWL_ENGINE, "/env/owl.engine")
        b = NanoOwlNanoSamBackend()
        assert (
            b._resolve_engine("/explicit/owl.engine", RES_NANOOWL_ENGINE, "owl")
            == "/explicit/owl.engine"
        )

    def test_resolve_engine_missing_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(RES_NANOOWL_ENGINE, raising=False)
        b = NanoOwlNanoSamBackend()
        with pytest.raises(RuntimeError, match="No TRT engine path"):
            b._resolve_engine(None, RES_NANOOWL_ENGINE, "NanoOWL image encoder")

    def test_segment_skips_without_deps(self) -> None:
        backend = NanoOwlNanoSamBackend()
        if not backend.is_available():
            pytest.skip("nanoowl/nanosam/trt deps not installed")
        # If deps ARE available we'd still need engines + GPU.

    def test_defaults(self) -> None:
        b = NanoOwlNanoSamBackend()
        assert b._owl_model_name == "google/owlvit-base-patch32"
        assert b._detection_threshold == 0.1
        assert b._device == "cuda"
        assert b._owl_engine_path is None
        assert b._sam_image_encoder_engine_path is None
        assert b._sam_mask_decoder_engine_path is None
