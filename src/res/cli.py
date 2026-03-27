from __future__ import annotations

from pathlib import Path

import click
import numpy as np
from PIL import Image

from res.api import segment
from res.backends import BACKEND_REGISTRY


@click.group()
def main() -> None:
    """RES -- Referring Expression Segmentation CLI."""


@main.command("segment")
@click.option("--image", required=True, type=click.Path(exists=True, dir_okay=False), help="Path to input image.")
@click.option("--prompt", required=True, type=str, help="Referring expression prompt.")
@click.option("--backend", default="mock", show_default=True, help="Backend name.")
@click.option("--output", required=True, type=click.Path(file_okay=False), help="Output directory for mask PNGs.")
def segment_cmd(image: str, prompt: str, backend: str, output: str) -> None:
    """Segment an image using a referring expression."""
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_img = np.array(Image.open(image).convert("RGB"))
    results = segment(src_img, prompt, backend=backend)

    click.echo(f"Found {len(results)} result(s):")
    for i, r in enumerate(results):
        click.echo(f"  [{i}] label={r.label!r}  score={r.score:.4f}  bbox={r.bbox}")

        # Binary grayscale mask
        mask_img = Image.fromarray(r.mask, mode="L")
        mask_path = out_dir / f"mask_{i}.png"
        mask_img.save(mask_path)

        # Alpha-blended overlay: original image visible only through mask
        overlay = Image.fromarray(src_img).convert("RGBA")
        alpha = Image.fromarray(r.mask, mode="L")
        overlay.putalpha(alpha)
        overlay_path = out_dir / f"overlay_{i}.png"
        overlay.save(overlay_path)

        click.echo(f"       mask   -> {mask_path}")
        click.echo(f"       overlay -> {overlay_path}")


@main.command("backends")
def backends_cmd() -> None:
    """List all registered backends and their availability."""
    for name in sorted(BACKEND_REGISTRY):
        cls = BACKEND_REGISTRY[name]
        available = cls().is_available()
        status = click.style("available", fg="green") if available else click.style("not available", fg="red")
        click.echo(f"  {name:20s} {status}")


@main.command("ui")
@click.option("--backend", default=None, help="Default backend for the UI.")
@click.option("--port", default=7860, show_default=True, type=int, help="Port for the Gradio server.")
def ui_cmd(backend: str | None, port: int) -> None:
    """Launch the Gradio interactive UI."""
    try:
        from res.ui import build_ui
    except ImportError:
        raise click.ClickException(
            "Gradio is not installed. Install with: uv pip install 'res[ui]'"
        )
    app = build_ui(default_backend=backend)
    app.launch(server_port=port, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
