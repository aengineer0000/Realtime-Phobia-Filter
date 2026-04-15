"""
Real-time Phobia Filter: Insect Detector Training Script
Optimized for Intel Arc GPU (XPU) via WSL
"""

import os
import wandb
import torch
import ultralytics.utils.torch_utils as torch_utils

# --- INTEL ARC (XPU) COMPATIBILITY PATCH ---
if torch.xpu.is_available():
    original_select_device = torch_utils.select_device
    torch_utils.select_device = lambda device="", *args, **kwargs: \
        torch.device('xpu') if str(device).lower() == 'xpu' else \
        original_select_device(device, *args, **kwargs)

    # 2. Fix Memory Management (stops the 'Torch not compiled with CUDA' error)
    # We point the hard-coded CUDA calls directly to XPU functions
    if not hasattr(torch, 'cuda') or not torch.cuda.is_available():
        class MockCuda:
            @staticmethod
            def get_device_properties(device): return torch.xpu.get_device_properties(device)
            @staticmethod
            def memory_reserved(device=None): return torch.xpu.memory_reserved(device)
            @staticmethod
            def memory_allocated(device=None): return torch.xpu.memory_allocated(device)
            @staticmethod
            def is_available(): return False
        
        torch.cuda.get_device_properties = MockCuda.get_device_properties
        torch.cuda.memory_reserved = MockCuda.memory_reserved
        torch.cuda.memory_allocated = MockCuda.memory_allocated
        print("--- INTEL ARC PATCHES ACTIVE: Memory & Device Redirection ---")
# --------------------------------------------

if torch.xpu.is_available():
    torch.cuda.get_device_properties = lambda device: torch.xpu.get_device_properties(device)
    print("XPU Memory Redirection Active")

from ultralytics import YOLOWorld

def main():
    wandb.init(
        project="phobia-detector", 
        name="insect_finetune",
        config={
            "architecture": "YOLO-World-Small",
            "dataset": "Roboflow-Insects",
            "device": "Intel Arc GPU"
        }
    )

    if torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"--- SUCCESS: Training on {torch.xpu.get_device_name(0)} ---")
    else:
        device = "cpu"
        print("--- WARNING: XPU not found, falling back to CPU ---")

    model = YOLOWorld('yolov8s-worldv2.pt')

    model.set_classes(["bug"])

    model.train(
        data='insectdata/data.yaml',    # Path to your roboflow data.yaml
        epochs=50,
        imgsz=640,                      # Standard for video inference
        device=device,                  # Pass the torch.device('xpu') object
        
        # --- STABILITY SETTINGS (Prevents VS Code/Kernel Crashes) ---
        batch=4,                        # Small batch size for laptop VRAM stability
        workers=0,                      # Fewer workers to prevent WSL process bloating
        amp=False,                      # Disable Mixed Precision for Intel stability
        patience=10,                     # Early stopping to prevent overfitting
        
        # --- THE STABILIZERS ---
        lr0=0.0001,         # Much lower starting point
        lrf=0.01,           # Final learning rate
        warmup_epochs=5.0,  # Slow start to let gradients settle
        weight_decay=0.01,  # Stronger penalty for "exploding" weights
        freeze=20,          # Protect the pre-trained features

        # --- AUGMENTATION (For Generalization) ---
        # --- SIMPLIFY FOR STABILITY ---
        mosaic=0.0,      # Turn off Mosaic for the first 10-20 epochs
        mixup=0.0,       # Turn off Mixup
        # Keep these (they are "Safe" augmentations)
        degrees=15.0,    
        scale=0.5,       
        fliplr=0.5,      
        flipud=0.2,      
        # ------------------------------
        close_mosaic=0,  # Since it's off, we don't need to close it
        
        # --- LOGGING ---
        project="phobia-detector",
        name="insect_finetune",
        exist_ok=True,
        plots=True                      # Generates training charts for your report
    )

    print("--- Training Complete. Optimizing for OpenVINO... ---")
    model.export(format='openvino', imgsz=640, half=True)
    
    print("--- PROCESS COMPLETE ---")
    print("Model weights and OpenVINO files are in the 'runs/detect/' folder.")
    wandb.finish()

if __name__ == "__main__":
    main()
