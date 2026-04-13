"""
app.py
------
Gradio web UI for the phobia content filter pipeline.
 
Run with:
    pip install gradio
    python app.py
"""
 
import gradio as gr
import tempfile
import os
from pipeline import process_video
 
 
# ── Gradio handler ─────────────────────────────────────────────────────────────
def run_pipeline(
    video_file,
    enable_trypophobia: bool,
    enable_insects: bool,
    trypo_threshold: float,
    blur_strength: int,
    frame_skip: int,
    progress=gr.Progress(),
):
    if video_file is None:
        raise gr.Error("Please upload a video file first.")
 
    if not enable_trypophobia and not enable_insects:
        raise gr.Error("Enable at least one detector.")
 
    # Write output
    output_path = os.path.join(os.path.dirname(video_file), "processed_output.mp4")
 
    def _progress(current, total):
        progress(current / max(total, 1), desc=f"Processing frame {current}/{total}")
 
    process_video(
        input_path          = video_file,
        output_path         = output_path,
        enable_trypophobia  = enable_trypophobia,
        enable_insects      = enable_insects,
        trypo_threshold     = trypo_threshold,
        blur_strength       = int(blur_strength),
        frame_skip          = int(frame_skip),
        progress_callback   = _progress,
    )
 
    return output_path
 
 
# ── UI layout ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Phobia Content Filter", theme=gr.themes.Soft()) as demo:
 
    gr.Markdown(
        """
        # 🎥 Phobia Content Filter
        Upload a video and select which content types to blur.
        Processed video will be available for download when complete.
        """
    )
 
    with gr.Row():
        # ── Left column: inputs ──────────────────────────────────────────────
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video", sources=["upload"])
 
            gr.Markdown("### Detectors")
            enable_trypo   = gr.Checkbox(label="🔴 Trypophobia (clusters/holes)", value=True)
            enable_insects = gr.Checkbox(label="🕷️  Insects / Spiders (YOLOWorld)", value=False)
 
            gr.Markdown("### Sensitivity")
            trypo_thresh = gr.Slider(
                minimum=0.3, maximum=0.99, value=0.7, step=0.01,
                label="Trypophobia confidence threshold",
                info="Lower = more sensitive (more false positives)"
            )
            blur_strength = gr.Slider(
                minimum=11, maximum=101, value=81, step=2,
                label="Blur strength",
                info="Higher = stronger blur (must be odd)"
            )
            frame_skip = gr.Slider(
                minimum=1, maximum=5, value=2, step=1,
                label="Frame skip",
                info="Run inference every N frames (higher = faster)"
            )
 
            run_btn = gr.Button("▶ Process Video", variant="primary")
 
        # ── Right column: output ─────────────────────────────────────────────
        with gr.Column(scale=1):
            video_output = gr.Video(label="Processed Video")
 
    run_btn.click(
        fn      = run_pipeline,
        inputs  = [
            video_input,
            enable_trypo,
            enable_insects,
            trypo_thresh,
            blur_strength,
            frame_skip,
        ],
        outputs = video_output,
    )
 
    gr.Markdown(
        """
        ---
        **Current App State:**
        - Trypophobia detector uses ResNet18
        - Insect detector uses YOLOWorld Small
        - Frame skip reuses the previous frame's boxes to speed up processing
        - Trypophobia model blurs the whole frame, Insect model can blur specific parts of the image
        """
    )
 
 
if __name__ == "__main__":
    demo.launch(share=False)   # set share=True to get a public link