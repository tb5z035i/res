from __future__ import annotations

import pytest

from res.backends.efficientsam3 import (
    EfficientSAM3Backend,
    RES_EFFICIENTSAM3_CHECKPOINT,
)


class TestEfficientSAM3Backend:
    def test_registered(self) -> None:
        from res.backends import BACKEND_REGISTRY

        assert "efficientsam3" in BACKEND_REGISTRY
        assert BACKEND_REGISTRY["efficientsam3"] is EfficientSAM3Backend

    def test_is_available_returns_bool(self) -> None:
        result = EfficientSAM3Backend().is_available()
        assert isinstance(result, bool)

    def test_defaults(self) -> None:
        b = EfficientSAM3Backend()
        assert b._device is None
        assert b._confidence_threshold == 0.5
        assert b._checkpoint_path is None
        assert b._backbone_type == "tinyvit"
        assert b._model_name == "11m"
        assert b._text_encoder_type == "MobileCLIP-S1"
        assert b._model is None
        assert b._processor is None

    def test_constructor_stores_params(self) -> None:
        b = EfficientSAM3Backend(
            device="cuda:1",
            confidence_threshold=0.3,
            checkpoint_path="/tmp/ckpt.pth",
            backbone_type="efficientvit",
            model_name="b2",
            text_encoder_type="MobileCLIP-S0",
        )
        assert b._device == "cuda:1"
        assert b._confidence_threshold == 0.3
        assert b._checkpoint_path == "/tmp/ckpt.pth"
        assert b._backbone_type == "efficientvit"
        assert b._model_name == "b2"
        assert b._text_encoder_type == "MobileCLIP-S0"
        assert b._model is None
        assert b._processor is None

    def test_resolve_checkpoint_explicit(self) -> None:
        b = EfficientSAM3Backend(checkpoint_path="/explicit/ckpt.pth")
        assert b._resolve_checkpoint() == "/explicit/ckpt.pth"

    def test_resolve_checkpoint_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(RES_EFFICIENTSAM3_CHECKPOINT, "/env/ckpt.pth")
        b = EfficientSAM3Backend()
        assert b._resolve_checkpoint() == "/env/ckpt.pth"

    def test_resolve_checkpoint_explicit_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(RES_EFFICIENTSAM3_CHECKPOINT, "/env/ckpt.pth")
        b = EfficientSAM3Backend(checkpoint_path="/explicit/ckpt.pth")
        assert b._resolve_checkpoint() == "/explicit/ckpt.pth"

    def test_resolve_checkpoint_none_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(RES_EFFICIENTSAM3_CHECKPOINT, raising=False)
        b = EfficientSAM3Backend()
        assert b._resolve_checkpoint() is None

    def test_text_encoder_type_none(self) -> None:
        b = EfficientSAM3Backend(text_encoder_type=None)
        assert b._text_encoder_type is None

    def test_backbone_variants(self) -> None:
        for btype, mname in [
            ("tinyvit", "5m"),
            ("tinyvit", "21m"),
            ("efficientvit", "b0"),
            ("efficientvit", "b1"),
            ("repvit", "m0_9"),
            ("repvit", "m2_3"),
        ]:
            b = EfficientSAM3Backend(backbone_type=btype, model_name=mname)
            assert b._backbone_type == btype
            assert b._model_name == mname

    def test_segment_skips_without_deps(self) -> None:
        backend = EfficientSAM3Backend()
        if not backend.is_available():
            pytest.skip("sam3/torch/torchvision/timm deps not installed")
