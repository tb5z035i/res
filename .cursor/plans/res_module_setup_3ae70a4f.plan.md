---
name: RES module setup
overview: Set up a Python package called `res` (Referring Expression Segmentation) with an extensible backend system, CLI, Gradio UI, and comprehensive tests, managed by uv.
todos:
  - id: init-uv
    content: Initialize uv project with src layout (uv init --lib), configure pyproject.toml with deps, optional deps, entry points, and build system
    status: in_progress
  - id: core-types
    content: Create src/res/types.py with SegmentationResult dataclass
    status: pending
  - id: backend-interface
    content: Create src/res/backends/base.py (Backend ABC) and __init__.py (registry)
    status: pending
  - id: mock-backend
    content: Implement MockBackend in src/res/backends/mock.py with deterministic mask generation
    status: pending
  - id: placeholder-backends
    content: Create placeholder backends (sam2, grounded_sam2, qwen3vl_sam2) that raise NotImplementedError
    status: pending
  - id: api-layer
    content: Create src/res/api.py with segment() function and src/res/__init__.py re-exports
    status: pending
  - id: cli
    content: Create src/res/cli.py with click-based CLI (segment + ui commands)
    status: pending
  - id: gradio-ui
    content: Create src/res/ui.py with Gradio interactive page
    status: pending
  - id: unit-tests
    content: Write unit tests for types, mock backend, api, and cli
    status: pending
  - id: smoke-tests
    content: Write smoke tests for end-to-end import + CLI round-trip
    status: pending
  - id: readme
    content: Write a concise README with install, usage (library + CLI + UI), and extending instructions
    status: pending
isProject: false
---

# RES Module -- Referring Expression Segmentation

## Project Structure

```
res/
├── pyproject.toml
├── uv.lock
├── README.md
├── src/
│   └── res/
│       ├── __init__.py          # Public API re-exports
│       ├── api.py               # segment() top-level function + backend registry
│       ├── types.py             # SegmentationResult dataclass
│       ├── backends/
│       │   ├── __init__.py      # Registry helpers
│       │   ├── base.py          # Abstract Backend interface
│       │   ├── mock.py          # MockBackend (deterministic, no deps)
│       │   ├── sam2.py          # SAM 2 placeholder
│       │   ├── grounded_sam2.py # Grounded-SAM-2 placeholder
│       │   └── qwen3vl_sam2.py  # Qwen3-VL + SAM2 placeholder
│       ├── cli.py               # CLI via click
│       └── ui.py                # Gradio interactive UI
├── tests/
│   ├── conftest.py              # Shared fixtures (sample images, etc.)
│   ├── unit/
│   │   ├── test_types.py
│   │   ├── test_mock_backend.py
│   │   ├── test_api.py
│   │   └── test_cli.py
│   └── smoke/
│       └── test_smoke.py        # End-to-end: CLI invocation, import, mock round-trip
```

## Core Types -- `src/res/types.py`

```python
@dataclasses.dataclass
class SegmentationResult:
    mask: np.ndarray                       # (H, W) binary, dtype=uint8 (0 or 255)
    score: float                           # confidence in [0, 1]
    label: str                             # matched label / phrase
    bbox: tuple[int, int, int, int]        # (x1, y1, x2, y2)
```

## Backend Interface -- `src/res/backends/base.py`

```python
class Backend(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray, prompt: str) -> list[SegmentationResult]: ...

    def is_available(self) -> bool:
        """Return True if all runtime deps for this backend are satisfied."""
        return True
```

All future backends (SAM 2, Grounded-SAM-2, Qwen3-VL+SAM2, etc.) subclass this. The interface is intentionally minimal so new backends only need to implement `segment()`.

## Backend Registry -- `src/res/backends/__init__.py` + `src/res/api.py`

Backends register by name via a simple dict registry:

```python
BACKEND_REGISTRY: dict[str, type[Backend]] = {}

def register_backend(name: str, cls: type[Backend]): ...
def get_backend(name: str, **kwargs) -> Backend: ...
```

Built-in backends (`mock`, `sam2`, `grounded_sam2`, `qwen3vl_sam2`) are registered at import time. The top-level public API:

```python
# src/res/api.py
def segment(
    image: np.ndarray | str | Path,
    prompt: str,
    backend: str = "mock",
    **backend_kwargs,
) -> list[SegmentationResult]:
```

Accepts a file path or ndarray for convenience.

## Mock Backend -- `src/res/backends/mock.py`

- No external dependencies beyond numpy/Pillow
- Generates a deterministic elliptical mask centered on the image
- Returns one `SegmentationResult` with `score=1.0` and `label=prompt`
- Used for testing the full pipeline without GPU/model dependencies

## Placeholder Backends

`sam2.py`, `grounded_sam2.py`, `qwen3vl_sam2.py` each define the class and register it, but raise `NotImplementedError` (or a custom `BackendNotAvailableError`) with a helpful message when `segment()` is called and deps are missing. This keeps the package importable regardless of which optional deps are installed.

## CLI -- `src/res/cli.py`

Uses `click`. Two commands exposed as console scripts (`res` entry point):

- `**res segment**` -- `--image PATH --prompt TEXT --backend NAME --output PATH`
Loads image, runs segmentation, saves each mask as a PNG (with alpha or grayscale). Prints summary (count, scores) to stdout.
- `**res ui**` -- `--backend NAME --port INT`
Launches the Gradio interface.

Entry point in `pyproject.toml`:

```toml
[project.scripts]
res = "res.cli:main"
```

## Gradio UI -- `src/res/ui.py`

- Image upload + text prompt input
- Backend selector dropdown (populated from registry, filtered to `is_available()`)
- Output: overlay visualization (original image + colored mask) and downloadable mask
- Simple single-page layout

Gradio is an optional dependency under `[project.optional-dependencies] ui = ["gradio>=5"]`.

## Dependency Strategy (`pyproject.toml`)

- **Core deps**: `numpy`, `Pillow`, `click`
- **Optional dep groups**:
  - `ui`: `gradio>=5`
  - `sam2`: (placeholder for future SAM 2 deps)
  - `grounded-sam2`: (placeholder)
  - `qwen3vl-sam2`: (placeholder)
  - `dev`: `pytest`, `pytest-cov`
  - `all`: union of all above

Build system: `hatchling` (well-supported by uv for `src/` layout).

## Tests

**Unit tests** (`tests/unit/`):

- `test_types.py` -- SegmentationResult construction, mask shape/dtype validation
- `test_mock_backend.py` -- MockBackend returns correct shapes, scores, handles edge cases (tiny images, empty prompts)
- `test_api.py` -- `segment()` dispatches to correct backend, handles path vs ndarray input, raises on unknown backend
- `test_cli.py` -- CLI invocation via `click.testing.CliRunner`, checks exit codes and output file creation

**Smoke tests** (`tests/smoke/`):

- `test_smoke.py` -- end-to-end: import `res`, call `segment()` with mock backend on a real test image, verify mask is valid ndarray; run CLI subprocess and check output file exists

## uv Project Init

```bash
uv init --lib --name res --package
```

This scaffolds `pyproject.toml` with `src/` layout. Then configure build-system, dependencies, and entry points manually.