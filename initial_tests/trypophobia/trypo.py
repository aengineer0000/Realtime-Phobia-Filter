import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path


# NEED TO MAKE A CONFIG FILE (JSON)

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH   = "trypophobia.csv" #path/to/dataset
IMG_SIZE   = 224 # Default
BATCH_SIZE = 32
EPOCHS     = 15 
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── 1. Dataset ───────────────────────────────────────────────────────────────
class TrypoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"])


# ── 2. Transforms ────────────────────────────────────────────────────────────
print("Transforming Data")
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── 3. Data splits ───────────────────────────────────────────────────────────
print("Reading Dataset")
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")
print(f"Train balance:\n{train_df['label'].value_counts()}\n")

train_ds = TrypoDataset(train_df, transform=train_transform)
val_ds   = TrypoDataset(val_df,   transform=val_transform)


# ── 4. Imbalance: WeightedRandomSampler ─────────────────────────────────────
print("Using WeightedRandomSampler/Dataloader")
counts = train_df["label"].value_counts().sort_index().values.astype(float)
class_weights = 1.0 / counts
sample_weights = train_df["label"].map({0: class_weights[0], 1: class_weights[1]}).values
sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE,
    sampler=sampler,          # replaces shuffle=True
    num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, pin_memory=True
)


# ── 5. Model (pretrained ResNet18, swapped classifier head) ──────────────────
print('Initializing RESNET18')
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)   # 2 classes: norm / trypo
model = model.to(DEVICE)


# ── 6. Imbalance: weighted loss ──────────────────────────────────────────────
loss_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=loss_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# ── 7. Train / eval functions ────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader.dataset), correct / len(loader.dataset), all_preds, all_labels


# ── 8. Training loop ─────────────────────────────────────────────────────────
best_val_loss = float("inf")
print('Starting training')
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, preds, labels = eval_epoch(model, val_loader, criterion)
    scheduler.step()

    print(f"Epoch {epoch:02d}/{EPOCHS} "
          f"| train loss: {train_loss:.4f} acc: {train_acc:.4f} "
          f"| val loss: {val_loss:.4f} acc: {val_acc:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("   ✅ Saved best model")


# ── 9. Final evaluation ───────────────────────────────────────────────────────
model.load_state_dict(torch.load("best_model.pth"))
_, _, preds, labels = eval_epoch(model, val_loader, criterion)

print("\n── Classification Report ──")
print(classification_report(labels, preds, target_names=["norm", "trypo"]))
print("── Confusion Matrix ──")
print(confusion_matrix(labels, preds))
