from __future__ import annotations

from typing import Any

from res.backends.base import Backend

BACKEND_REGISTRY: dict[str, type[Backend]] = {}
_BACKEND_INSTANCES: dict[str, Backend] = {}


def register_backend(name: str, cls: type[Backend]) -> None:
    """Register a backend class under *name*."""
    BACKEND_REGISTRY[name] = cls


def get_backend(name: str, **kwargs: Any) -> Backend:
    """Return a (cached) backend instance by *name*.

    Instances created without extra *kwargs* are cached so that expensive
    model loading only happens once.  Passing *kwargs* always creates a
    fresh instance.

    Raises ``KeyError`` if *name* is not registered.
    """
    if name not in BACKEND_REGISTRY:
        available = ", ".join(sorted(BACKEND_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown backend {name!r}. Available: {available}")

    if kwargs:
        return BACKEND_REGISTRY[name](**kwargs)

    if name not in _BACKEND_INSTANCES:
        _BACKEND_INSTANCES[name] = BACKEND_REGISTRY[name]()
    return _BACKEND_INSTANCES[name]


# ---------------------------------------------------------------------------
# Auto-register built-in backends on import
# ---------------------------------------------------------------------------
from res.backends.mock import MockBackend  # noqa: E402
from res.backends.sam2 import SAM2Backend  # noqa: E402
from res.backends.sam3 import SAM3Backend  # noqa: E402
from res.backends.grounded_sam2 import GroundedSAM2Backend  # noqa: E402
from res.backends.qwen3vl_sam2 import Qwen3VLSAM2Backend  # noqa: E402

register_backend("mock", MockBackend)
register_backend("sam2", SAM2Backend)
register_backend("sam3", SAM3Backend)
register_backend("grounded_sam2", GroundedSAM2Backend)
register_backend("qwen3vl_sam2", Qwen3VLSAM2Backend)
