from __future__ import annotations

import time

import numpy as np
from PIL import Image

from res.backends import get_backend, BACKEND_REGISTRY


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

    timings: list[str] = []

    t0 = time.perf_counter()
    engine = get_backend(backend_name)
    t1 = time.perf_counter()
    timings.append(f"Backend init: {t1 - t0:.3f}s")

    t2 = time.perf_counter()
    results = engine.segment(image, prompt)
    t3 = time.perf_counter()
    timings.append(f"Segmentation: {t3 - t2:.3f}s")

    if not results:
        summary = "No results found.\n" + "\n".join(timings)
        return None, None, summary

    best = max(results, key=lambda r: r.score)

    t4 = time.perf_counter()
    overlay = image.copy()
    colour = np.array([30, 144, 255], dtype=np.uint8)
    mask_bool = best.mask > 0
    overlay[mask_bool] = (
        overlay[mask_bool].astype(np.float32) * 0.5
        + colour.astype(np.float32) * 0.5
    ).astype(np.uint8)
    mask_pil = Image.fromarray(best.mask, mode="L")
    t5 = time.perf_counter()
    timings.append(f"Post-process: {t5 - t4:.3f}s")

    total = t5 - t0
    timings.append(f"**Total: {total:.3f}s**")

    result_info = (
        f"Found {len(results)} result(s) | "
        f"best: label={best.label!r}, score={best.score:.4f}, bbox={best.bbox}"
    )
    summary = result_info + "\n\n" + " | ".join(timings)

    return overlay, mask_pil, summary


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
                timing_output = gr.Markdown(label="Timing")

        run_btn.click(
            fn=_run,
            inputs=[img_input, prompt_input, backend_dropdown],
            outputs=[overlay_output, mask_output, timing_output],
        )

    return app
