"""Build TensorRT engines for the nanoowl_nanosam backend.

Usage::

    uv run python -m res.tools.build_nanoowl_nanosam_engines owl-engine \\
        --model-name /path/to/owlvit-base-patch32 \\
        -o data/engines/owl_image_encoder.engine

    uv run python -m res.tools.build_nanoowl_nanosam_engines sam-image-encoder \\
        --onnx data/resnet18_image_encoder.onnx \\
        -o data/engines/nanosam_image_encoder.engine

    uv run python -m res.tools.build_nanoowl_nanosam_engines sam-mask-decoder \\
        --onnx data/mobile_sam_mask_decoder.onnx \\
        -o data/engines/nanosam_mask_decoder.engine

    uv run python -m res.tools.build_nanoowl_nanosam_engines all \\
        --owl-model-name /path/to/owlvit-base-patch32 \\
        --sam-image-encoder-onnx data/resnet18_image_encoder.onnx \\
        --sam-mask-decoder-onnx data/mobile_sam_mask_decoder.onnx \\
        -o data/engines

All checkpoints / weights must be provided as local paths.

Requires ``trtexec`` on PATH (ships with TensorRT).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import click


def _trtexec() -> str:
    """Locate ``trtexec`` binary."""
    path = shutil.which("trtexec")
    if path:
        return path
    fallback = "/usr/src/tensorrt/bin/trtexec"
    if os.path.isfile(fallback):
        return fallback
    click.echo(
        "ERROR: trtexec not found on PATH and not at "
        f"{fallback}. Install TensorRT or add trtexec to PATH.",
        err=True,
    )
    sys.exit(1)


def _run(args: list[str]) -> None:
    cmd = [_trtexec(), *args]
    click.echo(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Build TensorRT engines for the nanoowl_nanosam backend."""


@cli.command()
@click.option("-o", "--output", required=True, type=click.Path(), help="Engine output path.")
@click.option("--model-name", required=True, type=str,
              help="Local path to OWL-ViT weights directory (or HuggingFace model id).")
@click.option("--fp16/--no-fp16", default=True, show_default=True)
@click.option("--onnx-opset", default=17, show_default=True, type=int)
def owl_engine(output: str, model_name: str, fp16: bool, onnx_opset: int) -> None:
    """Build OWL-ViT image-encoder TRT engine (NanoOWL).

    --model-name should point to a local directory containing the OWL-ViT
    weights (i.e. the output of ``transformers``
    ``OwlViTForObjectDetection.from_pretrained`` / ``save_pretrained``).
    """
    from nanoowl.owl_predictor import OwlPredictor

    click.echo(f"Loading {model_name} …")
    predictor = OwlPredictor(model_name=model_name)

    onnx_path = os.path.join(tempfile.mkdtemp(), "owl_image_encoder.onnx")
    click.echo(f"Exporting ONNX → {onnx_path}")
    predictor.export_image_encoder_onnx(onnx_path, onnx_opset=onnx_opset)

    s = predictor.image_size
    args = [f"--onnx={onnx_path}", f"--saveEngine={output}", f"--shapes=image:1x3x{s}x{s}"]
    if fp16:
        args.append("--fp16")
    _run(args)
    click.echo(f"Saved → {output}")


@cli.command()
@click.option("--onnx", required=True, type=click.Path(exists=True), help="Image-encoder ONNX.")
@click.option("-o", "--output", required=True, type=click.Path(), help="Engine output path.")
@click.option("--fp16/--no-fp16", default=True, show_default=True)
def sam_image_encoder(onnx: str, output: str, fp16: bool) -> None:
    """Build NanoSAM image-encoder TRT engine from ONNX."""
    args = [f"--onnx={onnx}", f"--saveEngine={output}"]
    if fp16:
        args.append("--fp16")
    _run(args)
    click.echo(f"Saved → {output}")


@cli.command()
@click.option("--onnx", "onnx_path", type=click.Path(), default=None,
              help="Pre-exported mask-decoder ONNX (skips export).")
@click.option("--checkpoint", type=click.Path(exists=True), default=None,
              help="mobile_sam.pt — required when --onnx is not given.")
@click.option("--model-type", default="vit_t", show_default=True)
@click.option("-o", "--output", required=True, type=click.Path(), help="Engine output path.")
def sam_mask_decoder(onnx_path: str | None, checkpoint: str | None, model_type: str, output: str) -> None:
    """Build NanoSAM mask-decoder TRT engine.

    Supply either --onnx (pre-exported) or --checkpoint (exports automatically).
    """
    if onnx_path is None:
        if checkpoint is None:
            raise click.UsageError("Provide --onnx (pre-exported) or --checkpoint (to export).")
        from nanosam.tools.export_sam_mask_decoder_onnx import run_export

        onnx_path = os.path.join(tempfile.mkdtemp(), "mask_decoder.onnx")
        click.echo(f"Exporting mask-decoder ONNX → {onnx_path}")
        run_export(model_type=model_type, checkpoint=checkpoint, output=onnx_path,
                   opset=16, return_single_mask=False)

    _run([
        f"--onnx={onnx_path}", f"--saveEngine={output}",
        "--minShapes=point_coords:1x1x2,point_labels:1x1",
        "--optShapes=point_coords:1x2x2,point_labels:1x2",
        "--maxShapes=point_coords:1x10x2,point_labels:1x10",
    ])
    click.echo(f"Saved → {output}")


@cli.command()
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--owl-model-name", required=True, type=str,
              help="Local path to OWL-ViT weights directory.")
@click.option("--sam-image-encoder-onnx", required=True, type=click.Path(exists=True))
@click.option("--sam-mask-decoder-onnx", type=click.Path(exists=True), default=None,
              help="Pre-exported ONNX. If omitted, --mobile-sam-checkpoint is required.")
@click.option("--mobile-sam-checkpoint", type=click.Path(exists=True), default=None)
@click.option("--fp16/--no-fp16", default=True, show_default=True)
@click.pass_context
def all(ctx: click.Context, output_dir: str, owl_model_name: str,
        sam_image_encoder_onnx: str, sam_mask_decoder_onnx: str | None,
        mobile_sam_checkpoint: str | None, fp16: bool) -> None:
    """Build all three engines into --output-dir."""
    os.makedirs(output_dir, exist_ok=True)

    click.echo("\n=== 1/3  OWL-ViT image encoder ===\n")
    ctx.invoke(owl_engine, output=os.path.join(output_dir, "owl_image_encoder.engine"),
               model_name=owl_model_name, fp16=fp16)

    click.echo("\n=== 2/3  NanoSAM image encoder ===\n")
    ctx.invoke(sam_image_encoder, onnx=sam_image_encoder_onnx,
               output=os.path.join(output_dir, "nanosam_image_encoder.engine"), fp16=fp16)

    click.echo("\n=== 3/3  NanoSAM mask decoder ===\n")
    ctx.invoke(sam_mask_decoder, onnx_path=sam_mask_decoder_onnx, checkpoint=mobile_sam_checkpoint,
               output=os.path.join(output_dir, "nanosam_mask_decoder.engine"))

    click.echo(f"\nAll engines written to {output_dir}/")


if __name__ == "__main__":
    cli()
