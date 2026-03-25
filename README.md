# RES -- Referring Expression Segmentation

Extensible Python package for referring expression segmentation with a pluggable backend system, CLI, and Gradio UI.

## Install

```bash
# Core (numpy, Pillow, click)
uv sync

# With Gradio UI
uv sync --extra ui

# With dev tools (pytest, coverage)
uv sync --extra dev

# Everything (runtime optional deps)
uv sync --all-extras
```

## Usage

### Python API

```python
import res

# From a numpy array
results = res.segment(image_array, "the red car", backend="mock")

# From a file path
results = res.segment("photo.jpg", "the person on the left", backend="mock")

for r in results:
    print(r.label, r.score, r.bbox)
    # r.mask is a (H, W) uint8 ndarray with values 0 or 255
```

### CLI

```bash
# Segment an image, save masks to a directory
res segment --image photo.jpg --prompt "the red car" --output ./masks/

# Launch the Gradio web UI
res ui --port 7860
```

The `segment` command produces two files per result in the output directory:
- `mask_N.png` -- binary grayscale mask
- `overlay_N.png` -- original image with the mask applied as alpha

### Gradio UI

```bash
res ui
```

Upload an image, enter a text prompt, pick a backend, and click **Segment**. The UI shows an overlay visualization and a downloadable mask.

## Available Backends

| Name | Status | Description |
|------|--------|-------------|
| `mock` | Ready | Deterministic elliptical mask (no GPU needed) |
| `sam3` | Ready | [SAM 3](https://github.com/facebookresearch/sam3) -- Segment Anything with Concepts |
| `sam2` | Placeholder | SAM 2 |
| `grounded_sam2` | Placeholder | Grounded-SAM-2 |
| `qwen3vl_sam2` | Placeholder | Qwen3-VL + SAM2 |

### SAM 3 Setup

SAM 3 is included as a git submodule under `third_party/sam3`. To use it:

```bash
# Clone with submodules (if you haven't already)
git submodule update --init --recursive

# Install SAM 3 and its deps (requires CUDA GPU + PyTorch >= 2.7)
uv pip install torch>=2.7 torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -e third_party/sam3
```

**Checkpoint configuration** -- set the `RES_SAM3_CHECKPOINT` env var to point at a local weights file:

```bash
export RES_SAM3_CHECKPOINT=/path/to/sam3.pt
```

If unset, SAM 3 auto-downloads from HuggingFace on first run (requires `huggingface-cli login`).

Then use it via the API or CLI:

```bash
uv run res segment --image photo.jpg --prompt "the red car" --backend sam3 --output ./out/
```

```python
import res
results = res.segment("photo.jpg", "the red car on the left", backend="sam3")
```

## Extending -- Adding a New Backend

1. Create a new file in `src/res/backends/`, e.g. `my_backend.py`.
2. Subclass `Backend` and implement `segment()`:

```python
from res.backends.base import Backend
from res.types import SegmentationResult

class MyBackend(Backend):
    def segment(self, image, prompt):
        # your implementation
        return [SegmentationResult(mask=..., score=..., label=..., bbox=...)]

    def is_available(self):
        # return False if optional deps are missing
        return True
```

3. Register it in `src/res/backends/__init__.py`:

```python
from res.backends.my_backend import MyBackend
register_backend("my_backend", MyBackend)
```

## Tests

```bash
uv run pytest tests/ -v
```
