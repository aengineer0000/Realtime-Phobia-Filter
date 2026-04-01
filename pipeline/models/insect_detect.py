"""
insect_detector.py
------------------
Inference wrapper for the YOLOWorld insect detector.
Matches the same interface as TrypophobiaDetector:
    detector = InsectDetector(model, threshold=0.2)
    boxes    = detector.detect(bgr_frame)  # list of (x1, y1, x2, y2)

Includes persistence logic so insects stay blurred for a few frames
after they leave the frame or are lost by the tracker.
"""

import cv2
import numpy as np
import collections

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 640
DEVICE     = "cpu"


# ── Core detector class ───────────────────────────────────────────────────────
class InsectDetector:
    """
    Wraps the YOLOWorld tracker with persistence logic.

    Parameters
    ----------
    model             : your already-loaded YOLOWorld model
    threshold         : detection confidence threshold
    persistence_limit : frames to keep blurring after an insect is lost
    padding           : pixels to expand each box (catches legs/motion)
    """

    def __init__(self, model, threshold: float = 0.2, persistence_limit: int = 5, padding: int = 20):
        self.model             = model
        self.threshold         = threshold
        self.persistence_limit = persistence_limit
        self.padding           = padding

        # Persistence state
        self.last_known_boxes: dict[int, np.ndarray]         = {}
        self.frames_since_seen: collections.defaultdict[int] = collections.defaultdict(int)

        print(f"[InsectDetector] Loaded — conf={threshold}, persistence={persistence_limit}, pad={padding}px")

    def detect(self, bgr_frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Parameters
        ----------
        bgr_frame : np.ndarray
            A single video frame in BGR format (as returned by cv2).

        Returns
        -------
        list of (x1, y1, x2, y2) tuples with padding applied.
        """
        h, w = bgr_frame.shape[:2]

        # ── Run tracker ───────────────────────────────────────────────────────
        results = self.model.track(
            bgr_frame, persist=True,
            conf=self.threshold, imgsz=IMG_SIZE, device=DEVICE
        )

        current_frame_ids = []

        for r in results:
            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids   = r.boxes.id.cpu().numpy().astype(int)

                for box, obj_id in zip(boxes, ids):
                    self.last_known_boxes[obj_id]  = box
                    self.frames_since_seen[obj_id] = 0
                    current_frame_ids.append(obj_id)

        # ── Build box list (live + persistent ghost boxes) ────────────────────
        output_boxes = []

        for obj_id in list(self.last_known_boxes.keys()):
            if self.frames_since_seen[obj_id] < self.persistence_limit:
                x1, y1, x2, y2 = map(int, self.last_known_boxes[obj_id])

                # Apply padding, clamped to frame dimensions
                x1 = max(0, x1 - self.padding)
                y1 = max(0, y1 - self.padding)
                x2 = min(w, x2 + self.padding)
                y2 = min(h, y2 + self.padding)

                output_boxes.append((x1, y1, x2, y2))

                # Increment counter for IDs not seen this frame
                if obj_id not in current_frame_ids:
                    self.frames_since_seen[obj_id] += 1
            else:
                # Clean up stale tracks
                del self.last_known_boxes[obj_id]
                del self.frames_since_seen[obj_id]

        return output_boxes

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def reset(self):
        """Clear persistence state — call this between videos."""
        self.last_known_boxes.clear()
        self.frames_since_seen.clear()