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
| `nanoowl_nanosam` | Ready | [NanoOWL](https://github.com/NVIDIA-AI-IOT/nanoowl) + [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam) -- TensorRT-accelerated open-vocabulary detection and segmentation |
| `efficientsam3` | Ready | [EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3) -- Stage 1 distilled lightweight encoders (TinyViT / EfficientViT / RepViT + MobileCLIP) |

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

### NanoOWL + NanoSAM Setup

This backend combines NanoOWL (open-vocabulary object detection via OWL-ViT) with
NanoSAM (lightweight SAM distillation) for TensorRT-accelerated referring
expression segmentation.

#### System Requirements

- NVIDIA GPU with CUDA support
- [TensorRT](https://developer.nvidia.com/tensorrt) installed (provides `trtexec`)
- `trtexec` on `PATH` (or at `/usr/src/tensorrt/bin/trtexec` on Jetson / NVIDIA containers)
- [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) (`pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt.git`)

#### Install

```bash
git submodule update --init --recursive

# Install the backend and all its deps (nanoowl, nanosam, tensorrt, torch2trt, etc.)
uv sync --extra nanoowl-nanosam
```

#### Prepare Checkpoints and ONNX Files

The backend needs three TensorRT engines. All weights / ONNX files must be
available locally before building engines.

| Artifact | What it is | How to obtain |
|----------|-----------|---------------|
| OWL-ViT weights dir | HuggingFace model directory (e.g. `owlvit-base-patch32/`) | `huggingface-cli download google/owlvit-base-patch32 --local-dir data/owlvit-base-patch32` |
| `resnet18_image_encoder.onnx` | NanoSAM image encoder | Download from the [NanoSAM releases](https://github.com/NVIDIA-AI-IOT/nanosam#usage) (Google Drive link in the NanoSAM README) |
| `mobile_sam_mask_decoder.onnx` | NanoSAM mask decoder | Download from same link, **or** export yourself (see below) |

If you want to export the mask decoder ONNX yourself, you need `mobile_sam.pt`
from the [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) project:

```bash
# Download the MobileSAM checkpoint
wget https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt \
    -P assets/

# Export the mask decoder to ONNX
uv run python -m nanosam.tools.export_sam_mask_decoder_onnx \
    --model-type=vit_t \
    --checkpoint=assets/mobile_sam.pt \
    --output=data/mobile_sam_mask_decoder.onnx
```

#### Build TRT Engines

A build tool is provided as a module and invoked via `uv run python -m`:

**Build all three engines at once:**

```bash
uv run python -m res.tools.build_nanoowl_nanosam_engines all \
    --owl-model-name data/owlvit-base-patch32 \
    --sam-image-encoder-onnx data/resnet18_image_encoder.onnx \
    --sam-mask-decoder-onnx data/mobile_sam_mask_decoder.onnx \
    -o data/engines
```

Or if you have `mobile_sam.pt` and want to export the decoder ONNX on the fly:

```bash
uv run python -m res.tools.build_nanoowl_nanosam_engines all \
    --owl-model-name data/owlvit-base-patch32 \
    --sam-image-encoder-onnx data/resnet18_image_encoder.onnx \
    --mobile-sam-checkpoint assets/mobile_sam.pt \
    -o data/engines
```

**Build engines individually:**

```bash
# OWL-ViT image encoder (pass local weights directory)
uv run python -m res.tools.build_nanoowl_nanosam_engines owl-engine \
    --model-name data/owlvit-base-patch32 \
    -o data/engines/owl_image_encoder.engine

# NanoSAM image encoder
uv run python -m res.tools.build_nanoowl_nanosam_engines sam-image-encoder \
    --onnx data/resnet18_image_encoder.onnx \
    -o data/engines/nanosam_image_encoder.engine

# NanoSAM mask decoder
uv run python -m res.tools.build_nanoowl_nanosam_engines sam-mask-decoder \
    --onnx data/mobile_sam_mask_decoder.onnx \
    -o data/engines/nanosam_mask_decoder.engine
```

Use a different local weights directory (e.g. `data/owlvit-base-patch16`) on
the `owl-engine` and `all` commands to change the OWL-ViT variant.

#### Configure Paths

Set these environment variables so the backend can find the model weights and
engines at runtime:

```bash
# Local OWL-ViT weights directory (avoids HuggingFace download)
export RES_NANOOWL_MODEL_NAME=data/owlvit-base-patch32

# TensorRT engine paths
export RES_NANOOWL_ENGINE=data/engines/owl_image_encoder.engine
export RES_NANOSAM_IMAGE_ENCODER_ENGINE=data/engines/nanosam_image_encoder.engine
export RES_NANOSAM_MASK_DECODER_ENGINE=data/engines/nanosam_mask_decoder.engine
```

If `RES_NANOOWL_MODEL_NAME` is not set, the backend falls back to downloading
`google/owlvit-base-patch32` from HuggingFace on first run.

Alternatively, pass the paths directly in the Python API:

```python
import res

results = res.segment(
    "photo.jpg",
    "the red car",
    backend="nanoowl_nanosam",
    owl_model_name="data/owlvit-base-patch32",
    owl_engine_path="data/engines/owl_image_encoder.engine",
    sam_image_encoder_engine_path="data/engines/nanosam_image_encoder.engine",
    sam_mask_decoder_engine_path="data/engines/nanosam_mask_decoder.engine",
)
```

#### Usage

```bash
# CLI
uv run res segment --image photo.jpg --prompt "the red car" \
    --backend nanoowl_nanosam --output ./out/
```

```python
import res
results = res.segment("photo.jpg", "the red car", backend="nanoowl_nanosam")
```

### EfficientSAM3 Setup

[EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3) distils the
SAM 3 vision and text encoders into lightweight backbones (Stage 1), producing
models that are much smaller and faster while sharing the same decoder and
processor pipeline. This backend uses the Stage 1 merged checkpoints.

#### Install

```bash
git submodule update --init --recursive

# Install EfficientSAM3 and its deps (requires CUDA GPU + PyTorch >= 2.5)
uv sync --extra efficientsam3
```

#### Download Checkpoint

Download a Stage 1 checkpoint from the
[EfficientSAM3 Model Zoo](https://github.com/SimonZeng7108/efficientsam3#efficientsam3-model-zoo--weight-release).
The recommended default is **TinyViT-11M + MobileCLIP-S1**:

```bash
# Via wget
wget -P data/ https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_tinyvit_11m_mobileclip_s1.pth

# Or via huggingface-cli
huggingface-cli download Simon7108528/EfficientSAM3 \
    stage1_all_converted/efficient_sam3_tinyvit_11m_mobileclip_s1.pth \
    --local-dir data/
```

Available model variants (all Stage 1):

| Backbone family | Sizes | Notes |
|-----------------|-------|-------|
| TinyViT | `5m`, `11m`, `21m` | Good accuracy/speed balance |
| EfficientViT | `b0`, `b1`, `b2` | Smallest (`b0` is 0.68M params) |
| RepViT | `m0_9`, `m1_1`, `m2_3` | Mobile-optimised |

Each backbone has variants with and without a distilled MobileCLIP text
encoder. Checkpoints **with** MobileCLIP in the filename (e.g.
`efficient_sam3_tinyvit_11m_mobileclip_s1.pth`) require setting
`text_encoder_type` accordingly; checkpoints **without** it use the full SAM 3
text encoder (`text_encoder_type=None`).

#### Configure Checkpoint Path

Set the `RES_EFFICIENTSAM3_CHECKPOINT` env var:

```bash
export RES_EFFICIENTSAM3_CHECKPOINT=data/efficient_sam3_tinyvit_11m_mobileclip_s1.pth
```

Or pass `checkpoint_path` directly in the Python API. If neither is set the
backend attempts to auto-download from HuggingFace.

#### Selecting a Model Variant

The default is `backbone_type="tinyvit"`, `model_name="11m"`,
`text_encoder_type="MobileCLIP-S1"`. Override via constructor kwargs:

```python
import res

results = res.segment(
    "photo.jpg",
    "the red car",
    backend="efficientsam3",
    backbone_type="efficientvit",
    model_name="b2",
    text_encoder_type="MobileCLIP-S1",
    checkpoint_path="data/efficient_sam3_efficientvit-b2_mobileclip_s1.pth",
)
```

#### Usage

```bash
# CLI
uv run res segment --image photo.jpg --prompt "the red car" \
    --backend efficientsam3 --output ./out/
```

```python
import res
results = res.segment("photo.jpg", "the red car", backend="efficientsam3")
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
