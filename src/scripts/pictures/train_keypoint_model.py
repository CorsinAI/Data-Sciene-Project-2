"""
train_keypoint_model.py — Train a hand keypoint regressor on dataset1 (COCO annotations).

═══════════════════════════════════════════════════════════════════════════════════════
WHY THIS SCRIPT EXISTS
═══════════════════════════════════════════════════════════════════════════════════════
MediaPipe provides a pre-trained model that can detect 21 hand landmarks from any image.
To compare it fairly with a *custom* approach, we need our own model that does the same
job: given a raw image, predict the (x, y) coordinates of the 21 hand landmarks.

Dataset1 (hand_keypoint_dataset_26k) provides exactly the training data we need:
~18 k images each annotated with 21 COCO keypoints.  We use this to train a regression
model — the "custom keypoint extractor" — whose outputs will later be fed to the same
ASL-letter classifier as MediaPipe's outputs, making the comparison apples-to-apples.

═══════════════════════════════════════════════════════════════════════════════════════
ARCHITECTURE CHOICES
═══════════════════════════════════════════════════════════════════════════════════════
• Backbone: ResNet18 pretrained on ImageNet.
  WHY: We only have ~18 k images — too few to train a deep CNN from scratch without
  overfitting.  ImageNet pretraining gives us strong low-level feature detectors
  (edges, textures) for free; we only need to specialise the top layers.

• Head: Linear(512 → 42) — outputs the 42 raw (x, y) coordinates.
  WHY: Keypoint detection is a regression problem (continuous coordinates), not
  classification.  A single linear layer on top of the global-average-pooled ResNet
  features is sufficient and easy to interpret.

• Loss: MSE between predicted and ground-truth coordinates (both normalised to [0,1]).
  WHY: MSE directly penalises Euclidean distance in coordinate space, which maps
  cleanly to "how far off is each landmark?".  Mean Absolute Error is an alternative
  but MSE penalises large errors more heavily, which helps the model avoid "lazy"
  predictions near the mean.

═══════════════════════════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════════════════════════
  checkpoints/hand_keypoint_regressor.pt

USAGE
═══════════════════════════════════════════════════════════════════════════════════════
  python -m src.scripts.pictures.train_keypoint_model
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET1_ROOT = (
    PROJECT_ROOT
    / "data" / "dataset1"
    / "hand_keypoint_dataset_26k"
    / "hand_keypoint_dataset_26k"
)
TRAIN_JSON = DATASET1_ROOT / "coco_annotation" / "train" / "_annotations.coco.json"
VAL_JSON   = DATASET1_ROOT / "coco_annotation" / "val"   / "_annotations.coco.json"
TRAIN_IMG  = DATASET1_ROOT / "images" / "train"
VAL_IMG    = DATASET1_ROOT / "images" / "val"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "hand_keypoint_regressor.pt"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 128
NUM_EPOCHS  = 20
LR          = 1e-4
EARLY_STOP_PATIENCE = 4   # stop if val loss doesn't improve for this many epochs
SEED        = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def parse_coco_keypoints(json_path: Path, img_dir: Path):
    """
    Parse a COCO keypoint JSON and return a list of (image_path, kp_array) pairs.

    WHY: The COCO format stores keypoints as a flat list [x, y, v, x, y, v, ...]
    where v is visibility (0=not labelled, 1=occluded, 2=visible).  We discard v
    and keep only x, y.  Coordinates are normalised to [0, 1] by dividing by the
    image width/height so the model's output is always in the same range regardless
    of the original image resolution.

    Returns: list of (Path, np.ndarray shape (42,) float32)
    """
    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)

    # Build image_id → filename map
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_wh   = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

    samples = []
    for ann in coco["annotations"]:
        if ann.get("category_id") != 1:
            # category 0 = bounding box only, category 1 = keypoints
            continue
        raw_kps = ann["keypoints"]   # length 63: [x, y, v] × 21
        img_id  = ann["image_id"]
        w, h    = id_to_wh[img_id]
        fname   = id_to_file[img_id]

        # Extract x, y and normalise to [0, 1]
        xy = []
        for i in range(0, len(raw_kps), 3):
            x_norm = raw_kps[i]   / w
            y_norm = raw_kps[i+1] / h
            xy.extend([x_norm, y_norm])

        kp_array = np.array(xy, dtype=np.float32)  # (42,)
        img_path = img_dir / fname
        if img_path.exists():
            samples.append((img_path, kp_array))

    return samples


class KeypointDataset(Dataset):
    """
    PyTorch Dataset for keypoint regression.

    WHY AUGMENTATION:
    • Horizontal flip — hands appear in both orientations in the wild.  The
      keypoints are mirrored according to the MediaPipe/COCO hand topology so
      the labels stay correct after flipping.
    • Colour jitter — makes the model robust to different lighting conditions,
      skin tones, and camera white-balance, which vary significantly in our
      custom dataset2.

    WHY NOT MORE AUGMENTATION (rotation, crop):
    • Random rotations would require rotating keypoint coordinates too — easy but
      adds complexity.  For a keypoint detector, colour jitter + flip already
      covers the most common real-world variability.
    • We avoid heavy spatial augmentation because dataset1 images are already
      diverse in scale and framing.
    """

    # Landmark indices that must be swapped on a horizontal flip.
    # Hand topology: left thumb↔right thumb, etc.
    # For a single-hand dataset normalised to [0,1] width, flipping x is enough;
    # no index swapping needed because both hands share the same 21-point schema.
    FLIP_IDX = list(range(21))  # identity — no swap needed for single-hand data

    def __init__(self, samples, augment: bool = False):
        self.samples = samples
        self.augment  = augment

        self.colour_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, kp = self.samples[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        kp = kp.copy()

        if self.augment and random.random() < 0.5:
            # Horizontal flip: mirror x-coordinates (they're in [0,1])
            img = img[:, ::-1, :].copy()
            for i in range(0, 42, 2):
                kp[i] = 1.0 - kp[i]

        # To tensor + ImageNet normalisation
        x = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = torch.from_numpy(x).permute(2, 0, 1)  # (3, H, W)

        return x, torch.from_numpy(kp)


# ═════════════════════════════════════════════════════════════════════════════
# 2. MODEL
# ═════════════════════════════════════════════════════════════════════════════

class KeypointRegressor(nn.Module):
    """
    ResNet18 backbone with a linear regression head for 21 hand keypoints.

    WHY ResNet18:
    Lightweight enough to train on CPU in reasonable time (~20 min for 20 epochs),
    but deep enough to learn meaningful spatial features.  ResNet50 would give
    slightly better accuracy but at 4× the compute cost — not worth it here since
    the downstream classifier only needs rough landmark positions, not sub-pixel
    precision.

    WHY freeze backbone initially:
    The first few epochs only train the head on top of frozen ImageNet features.
    This "warm-up" prevents the randomly-initialised head from sending large
    gradients into the backbone and destroying the pretrained weights before the
    head has had a chance to learn.  After warm-up, the full network fine-tunes.
    """

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feature_dim = backbone.fc.in_features          # 512 for ResNet18
        backbone.fc = nn.Identity()                    # remove classification head
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, 42)         # predict 21 (x, y) pairs

    def forward(self, x):
        features = self.backbone(x)    # (B, 512)
        return self.head(features)     # (B, 42)


# ═════════════════════════════════════════════════════════════════════════════
# 3. TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def train():
    torch.manual_seed(SEED)
    random.seed(SEED)

    print(f"Device: {DEVICE}")
    print("Parsing dataset1 annotations…")

    train_samples = parse_coco_keypoints(TRAIN_JSON, TRAIN_IMG)
    val_samples   = parse_coco_keypoints(VAL_JSON,   VAL_IMG)

    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val   samples: {len(val_samples)}")

    train_ds = KeypointDataset(train_samples, augment=True)
    val_ds   = KeypointDataset(val_samples,   augment=False)

    # num_workers=0 avoids Windows multiprocessing spawn deadlocks when running via -m
    # pin_memory=True speeds up CPU→GPU transfers even with num_workers=0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = KeypointRegressor().to(DEVICE)
    criterion = nn.MSELoss()

    # ── Phase 1: head warm-up (backbone frozen) ────────────────────────────
    # WHY: see docstring on KeypointRegressor above.
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimiser = torch.optim.Adam(model.head.parameters(), lr=LR * 10)

    print("\n── Phase 1: head warm-up (2 epochs, backbone frozen) ──")
    for epoch in range(2):
        model.train()
        train_loss = _run_epoch(model, train_loader, criterion, optimiser, train=True)
        print(f"  Epoch {epoch+1}/2  train_loss={train_loss:.6f}")

    # ── Phase 2: full fine-tune ────────────────────────────────────────────
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=2, factor=0.5)

    print(f"\n── Phase 2: full fine-tune ({NUM_EPOCHS} epochs) ──")

    best_val_loss = float("inf")
    no_improve    = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = _run_epoch(model, train_loader, criterion, optimiser, train=True)

        model.eval()
        val_loss = _run_epoch(model, val_loader, criterion, optimiser=None, train=False)

        scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        marker = " ✓" if improved else ""
        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS}  train={train_loss:.6f}  val={val_loss:.6f}{marker}")

        if improved:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    print(f"\nBest val MSE : {best_val_loss:.6f}")
    print(f"Checkpoint  : {CHECKPOINT_PATH}")


def _run_epoch(model, loader, criterion, optimiser, train: bool) -> float:
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for imgs, kps in loader:
            imgs = imgs.to(DEVICE)
            kps  = kps.to(DEVICE)
            preds = model(imgs)
            loss  = criterion(preds, kps)
            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train()
