"""
pipeline.py
-----------
Core video processing pipeline.
 
Takes a video file path, runs enabled detectors on each frame,
blurs detected regions, and writes an output video.
 
Detectors used:
  - TrypophobiaDetector  (ResNet18, full-frame box)
  - YOLOWorldDetector    (stub — swap in your real model)
 
Both detectors share the same interface:
    boxes = detector.detect(bgr_frame)  ->  list of (x1, y1, x2, y2)
"""
 
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLOWorld
from models.trypophobia_detect import TrypophobiaDetector
from models.insect_detect import InsectDetector
 
  
# ── Blur helpers ───────────────────────────────────────────────────────────────
def blur_boxes(
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    blur_strength: int = 51,
) -> np.ndarray:
    """
    Apply Gaussian blur to every bounding box region in `frame`.
 
    Parameters
    ----------
    frame         : BGR frame from cv2
    boxes         : list of (x1, y1, x2, y2)
    blur_strength : kernel size (must be odd); higher = more blurred
    """
    for (x1, y1, x2, y2) in boxes:
        # Clamp to frame dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame[y1:y2, x1:x2]
        # Ensure kernel is odd
        k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return frame
 
 
# ── Main pipeline ──────────────────────────────────────────────────────────────
def process_video(
    input_path: str,
    output_path: str,
    enable_trypophobia: bool = True,
    enable_insects: bool     = False,
    trypo_threshold: float   = 0.7,
    blur_strength: int       = 81,
    frame_skip: int          = 2,
    progress_callback=None,
) -> str:
    """
    Process a video file through the phobia detection pipeline.
 
    Parameters
    ----------
    input_path          : path to the source video
    output_path         : path to write the blurred video
    enable_trypophobia  : run trypophobia detector
    enable_insects      : run YOLOWorld insect detector
    trypo_threshold     : confidence threshold for trypophobia (0–1)
    blur_strength       : Gaussian blur kernel size (odd int)
    frame_skip          : run inference every N frames (1 = every frame)
    progress_callback   : optional fn(current_frame, total_frames)
 
    Returns
    -------
    output_path on success
    """
 
    # ── Load detectors ──────────────────────────────────────────────────────
    detectors = []
 
    if enable_trypophobia:
        trypo = TrypophobiaDetector("weights/trypo.pth", threshold=trypo_threshold)
        detectors.append(trypo)
 
    if enable_insects:
        yolo_model = YOLOWorld("weights/insects.pt")
        # yolo_model.set_classes(["insect", "spider", "snake"])
        detectors.append(InsectDetector(yolo_model, threshold=0.2))
 
    if not detectors:
        raise ValueError("At least one detector must be enabled.")
    
    # Reset detectors
    for detector in detectors:
        if hasattr(detector, "reset"):
            detector.reset()
 
    # ── Open video ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_path}")
 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    # ── Writer ──────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
 
    print(f"[Pipeline] {total_frames} frames @ {fps:.1f} fps  |  {width}×{height}")
    print(f"[Pipeline] Detectors: {[type(d).__name__ for d in detectors]}")
    print(f"[Pipeline] Frame skip: every {frame_skip} frame(s)")
 
    # ── Frame loop ──────────────────────────────────────────────────────────
    frame_idx   = 0
    cached_boxes: list[tuple[int, int, int, int]] = []
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        # Run inference only on every Nth frame; reuse boxes in between
        if frame_idx % frame_skip == 0:
            cached_boxes = []
            for detector in detectors:
                cached_boxes.extend(detector.detect(frame))
 
        # Apply blur with cached boxes
        if cached_boxes:
            frame = blur_boxes(frame, cached_boxes, blur_strength=blur_strength)
 
        writer.write(frame)
 
        frame_idx += 1
        if progress_callback:
            progress_callback(frame_idx, total_frames)
 
        if frame_idx % 100 == 0:
            pct = 100 * frame_idx / max(total_frames, 1)
            print(f"  [{pct:5.1f}%] frame {frame_idx}/{total_frames}")
 
    cap.release()
    writer.release()
    print(f"[Pipeline] ✅ Done → {output_path}")
    return output_path
 
 
# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="Phobia content filter pipeline")
    parser.add_argument("input",  help="Path to input video")
    parser.add_argument("output", help="Path to output video")
    parser.add_argument("--no-trypo",   action="store_true", help="Disable trypophobia detector")
    parser.add_argument("--no-insects", action="store_true", help="Disable insect detector")
    parser.add_argument("--threshold",  type=float, default=0.7, help="Trypophobia confidence threshold")
    parser.add_argument("--blur",       type=int,   default=51,  help="Blur kernel size (odd int)")
    parser.add_argument("--skip",       type=int,   default=2,   help="Run inference every N frames")
    args = parser.parse_args()
 
    process_video(
        input_path          = args.input,
        output_path         = args.output,
        enable_trypophobia  = not args.no_trypo,
        enable_insects      = not args.no_insects,
        trypo_threshold     = args.threshold,
        blur_strength       = args.blur,
        frame_skip          = args.skip,
    )