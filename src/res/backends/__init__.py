from __future__ import annotations

from typing import Any

from res.backends.base import Backend

BACKEND_REGISTRY: dict[str, type[Backend]] = {}


def register_backend(name: str, cls: type[Backend]) -> None:
    """Register a backend class under *name*."""
    BACKEND_REGISTRY[name] = cls


def get_backend(name: str, **kwargs: Any) -> Backend:
    """Instantiate a registered backend by *name*.

    Raises ``KeyError`` if *name* is not registered.
    """
    if name not in BACKEND_REGISTRY:
        available = ", ".join(sorted(BACKEND_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown backend {name!r}. Available: {available}")
    return BACKEND_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Auto-register built-in backends on import
# ---------------------------------------------------------------------------
from res.backends.mock import MockBackend  # noqa: E402
from res.backends.sam2 import SAM2Backend  # noqa: E402
from res.backends.grounded_sam2 import GroundedSAM2Backend  # noqa: E402
from res.backends.qwen3vl_sam2 import Qwen3VLSAM2Backend  # noqa: E402

register_backend("mock", MockBackend)
register_backend("sam2", SAM2Backend)
register_backend("grounded_sam2", GroundedSAM2Backend)
register_backend("qwen3vl_sam2", Qwen3VLSAM2Backend)
