import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "best_model.pth"
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS     = {0: "norm", 1: "trypo"}


# ── Rebuild model architecture & load weights ──────────────────────────────
def load_model(path):
    model = models.resnet18(weights=None)           # same architecture as training
    model.fc = nn.Linear(model.fc.in_features, 2)  # same head as training
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # disables dropout/batchnorm training behaviour
    return model

model = load_model(MODEL_PATH)


# ── 2. Same transforms as validation (no augmentation) ───────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Predict a single image ─────────────────────────────────────────────────
def predict_image(img_path, model):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)  # add batch dim

    with torch.no_grad():
        out  = model(tensor)
        prob = torch.softmax(out, dim=1)[0]
        pred = prob.argmax().item()

    print(f"Prediction : {LABELS[pred]}")
    print(f"Confidence : {prob[pred]:.2%}")
    print(f"  norm  (0): {prob[0]:.2%}")
    print(f"  trypo (1): {prob[1]:.2%}")
    return pred, prob


# ── Predict a whole folder ─────────────────────────────────────────────────
def predict_folder(folder_path, model):
    from pathlib import Path
    results = []
    for img_path in Path(folder_path).rglob("*.png"):
        pred, prob = predict_image(str(img_path), model)
        results.append({
            "filepath": str(img_path),
            "prediction": LABELS[pred],
            "confidence": f"{prob[pred]:.2%}"
        })

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("predictions.csv", index=False)
    print(f"\n✅ Saved predictions for {len(df)} images to predictions.csv")
    return df


if __name__ == "__main__":
    # Single image
    predict_image("bicycle.jpg", model)

    # Whole folder
    # predict_folder("path/to/folder", model)


    '''
    Notes: Add command-line arguments, and code to check whether its a stream, img or
    directory
    
    '''
