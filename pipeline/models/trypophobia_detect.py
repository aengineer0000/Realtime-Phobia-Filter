"""
trypophobia_detector.py
-----------------------
Inference wrapper for the trained ResNet18 trypophobia classifier.
Returns a bounding box covering the full frame when the model fires,
so it plugs into the same box-based blur pipeline as YOLOWorld.
"""
 
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
 
 
# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
 
 
# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(weights_path: str) -> nn.Module:
    """Load the trained ResNet18 and move it to the correct device."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"[TrypophobiaDetector] Loaded weights from '{weights_path}' on {DEVICE}")
    return model
 
 
# ── Core detector class ───────────────────────────────────────────────────────
class TrypophobiaDetector:
    """
    Wraps the ResNet18 classifier so it speaks the same interface as
    a box-based detector.
 
    Usage:
        detector = TrypophobiaDetector("best_model.pth", threshold=0.7)
        boxes    = detector.detect(bgr_frame)   # list of (x1,y1,x2,y2) or []
    """
 
    def __init__(self, weights_path: str, threshold: float = 0.7):
        self.model     = load_model(weights_path)
        self.threshold = threshold
 
    def detect(self, bgr_frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Parameters
        ----------
        bgr_frame : np.ndarray
            A single video frame in BGR format (as returned by cv2).
 
        Returns
        -------
        list of (x1, y1, x2, y2) tuples.
        If trypophobic content is detected, returns one box covering the
        entire frame.  Otherwise returns an empty list.
        """
        # cv2 BGR → PIL RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
 
        tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)
 
        with torch.no_grad():
            logits = self.model(tensor)                        # shape (1, 2)
            probs  = torch.softmax(logits, dim=1)
            trypo_prob = probs[0, 1].item()                    # class 1 = trypo
 
        if trypo_prob >= self.threshold:
            h, w = bgr_frame.shape[:2]
            return [(0, 0, w, h)]                              # full-frame box
        return []
 
    def set_threshold(self, threshold: float):
        self.threshold = threshold