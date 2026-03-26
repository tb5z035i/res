from __future__ import annotations

import colorsys
import time

import numpy as np

from res.backends import get_backend, BACKEND_REGISTRY


def _score_to_color(rank: int, total: int) -> tuple[int, int, int]:
    """Map rank (0 = highest score) to a rainbow color from red to blue."""
    if total <= 1:
        hue = 0.0
    else:
        hue = rank / (total - 1) * 0.667  # 0.0 (red) -> 0.667 (blue)
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


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
        return None, "", summary

    t4 = time.perf_counter()
    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    n = len(ranked)
    colors = [_score_to_color(i, n) for i in range(n)]

    overlay = image.copy()
    for (r, g, b), result in zip(reversed(colors), reversed(ranked)):
        overlay[result.mask > 0] = np.array([r, g, b], dtype=np.uint8)
    t5 = time.perf_counter()
    timings.append(f"Post-process: {t5 - t4:.3f}s")

    total = t5 - t0
    timings.append(f"**Total: {total:.3f}s**")

    lines = [f"Found {n} result(s)"]
    for i, (result, (r, g, b)) in enumerate(zip(ranked, colors)):
        lines.append(
            f"- **#{i + 1}** label={result.label!r}, score={result.score:.4f}, "
            f"bbox={result.bbox}, color=rgb({r},{g},{b})"
        )
    summary = "\n".join(lines) + "\n\n" + " | ".join(timings)

    high = ranked[0].score
    low = ranked[-1].score
    color_bar = (
        '<div style="display:flex;align-items:center;gap:6px;margin:4px 0">'
        f'<span style="font-size:12px">{high:.4f}</span>'
        '<div style="flex:1;height:14px;border-radius:3px;'
        "background:linear-gradient(to right,"
        "hsl(0,100%,50%),hsl(60,100%,50%),hsl(120,100%,50%),"
        'hsl(180,100%,50%),hsl(240,100%,50%))"></div>'
        f'<span style="font-size:12px">{low:.4f}</span>'
        "</div>"
    )

    return overlay, color_bar, summary


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
                color_bar_output = gr.HTML()
                timing_output = gr.Markdown(label="Info")

        run_btn.click(
            fn=_run,
            inputs=[img_input, prompt_input, backend_dropdown],
            outputs=[overlay_output, color_bar_output, timing_output],
        )

    return app
