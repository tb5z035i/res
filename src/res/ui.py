from __future__ import annotations

import numpy as np
from PIL import Image

from res.api import segment
from res.backends import BACKEND_REGISTRY


def _available_backends() -> list[str]:
    return [
        name
        for name, cls in sorted(BACKEND_REGISTRY.items())
        if cls().is_available()
    ]


def _run(image: np.ndarray | None, prompt: str, backend_name: str):
    """Callback for the Gradio interface."""
    if image is None:
        raise ValueError("Please upload an image.")
    if not prompt.strip():
        raise ValueError("Please enter a text prompt.")

    results = segment(image, prompt, backend=backend_name)
    if not results:
        return None, None

    best = max(results, key=lambda r: r.score)

    # Overlay: original with coloured mask blended on top
    h, w = image.shape[:2]
    overlay = image.copy()
    colour = np.array([30, 144, 255], dtype=np.uint8)  # dodger-blue
    mask_bool = best.mask > 0
    overlay[mask_bool] = (
        overlay[mask_bool].astype(np.float32) * 0.5
        + colour.astype(np.float32) * 0.5
    ).astype(np.uint8)

    mask_pil = Image.fromarray(best.mask, mode="L")
    return overlay, mask_pil


def build_ui(default_backend: str | None = None):
    """Construct and return the Gradio Blocks app (does not call .launch())."""
    import gradio as gr

    backends = _available_backends()
    if not backends:
        backends = ["mock"]
    default = default_backend if default_backend in backends else backends[0]

    with gr.Blocks(title="RES -- Referring Expression Segmentation") as app:
        gr.Markdown("## RES -- Referring Expression Segmentation")

        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="numpy", label="Input Image")
                prompt_input = gr.Textbox(label="Prompt", placeholder="e.g. the red car on the left")
                backend_dropdown = gr.Dropdown(
                    choices=backends,
                    value=default,
                    label="Backend",
                )
                run_btn = gr.Button("Segment", variant="primary")

            with gr.Column():
                overlay_output = gr.Image(type="numpy", label="Overlay")
                mask_output = gr.Image(type="pil", label="Mask (downloadable)")

        run_btn.click(
            fn=_run,
            inputs=[img_input, prompt_input, backend_dropdown],
            outputs=[overlay_output, mask_output],
        )

    return app
