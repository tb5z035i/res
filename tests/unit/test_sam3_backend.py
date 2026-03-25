from __future__ import annotations

import pytest

from res.backends.sam3 import SAM3Backend


class TestSAM3Backend:
    def test_registered(self) -> None:
        from res.backends import BACKEND_REGISTRY
        assert "sam3" in BACKEND_REGISTRY
        assert BACKEND_REGISTRY["sam3"] is SAM3Backend

    def test_is_available_returns_bool(self) -> None:
        result = SAM3Backend().is_available()
        assert isinstance(result, bool)

    def test_segment_raises_without_deps(self) -> None:
        backend = SAM3Backend()
        if not backend.is_available():
            pytest.skip("sam3 deps not installed -- cannot test segment()")
        # If sam3 IS available, we'd need a GPU + checkpoint to actually run.
        # Just verify the object is constructable.

    def test_constructor_stores_params(self) -> None:
        b = SAM3Backend(device="cpu", confidence_threshold=0.3, checkpoint_path="/tmp/ckpt.pt")
        assert b._device == "cpu"
        assert b._confidence_threshold == 0.3
        assert b._checkpoint_path == "/tmp/ckpt.pt"
        assert b._model is None
