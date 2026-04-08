"""
utils.py — Shared helpers for the picture-based ASL classification pipeline.

WHY THIS FILE EXISTS:
Both train_keypoint_model.py and classify_and_compare.py need the same data-loading,
normalisation, and visualisation logic. Centralising it here means a single source of truth:
change the normalisation formula once, it applies to both pipelines automatically. It also keeps
the main scripts focused on their own logic rather than boilerplate.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Paths ────────────────────────────────────────────────────────────────────
# Resolve from this file's location so scripts can be run from any working dir.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET2_DIR = PROJECT_ROOT / "data" / "dataset2" / "own_dataset"

# ImageNet normalisation constants (used by both pipelines because even MediaPipe
# produces raw pixel images that we pass through the ResNet backbone).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = 224  # ResNet18 canonical input size


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_image_rgb(path: str | Path) -> np.ndarray:
    """
    Load an image, convert to RGB, and resize to IMG_SIZE × IMG_SIZE.

    WHY: OpenCV reads BGR by default; every downstream consumer (torch models,
    MediaPipe) expects RGB.  Centralising the resize avoids mismatched sizes
    between the training and inference paths.

    Returns: uint8 np.ndarray of shape (224, 224, 3).
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def image_to_tensor(img: np.ndarray) -> "torch.Tensor":
    """
    Normalise a (H, W, 3) uint8 RGB array and return a (3, H, W) float32 tensor.

    WHY: The ResNet18 backbone we use was pretrained on ImageNet; applying the
    same mean/std normalisation at inference time keeps the activations in the
    distribution the backbone was optimised for.
    """
    import torch
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD          # (H, W, 3)
    return torch.from_numpy(x).permute(2, 0, 1)      # (3, H, W)


# ── Keypoint normalisation ────────────────────────────────────────────────────

def normalise_keypoints(kps: np.ndarray) -> np.ndarray:
    """
    Make a (42,) flat keypoint vector invariant to hand position and size.

    Procedure:
        1. Reshape to (21, 2).
        2. Translate so that landmark 0 (wrist) sits at the origin.
        3. Scale so that the distance from wrist to middle-finger MCP
           (landmark 9) equals 1.  This is the most stable reference distance
           because it is always visible and roughly constant across hand poses.
        4. Flatten back to (42,).

    WHY: Without normalisation a tiny hand in the centre and a large hand in the
    corner produce completely different feature vectors for the same sign. The
    classifier would waste capacity learning position/scale instead of shape.

    WHY landmark 9 as scale reference: it is always detected (not occluded by
    finger bending), it lies on the palm axis, and the wrist→MCP9 distance is
    proportional to overall hand size regardless of finger spread.

    Returns: (42,) float32 array with wrist at origin and wrist→MCP9 = 1.
    """
    kps = kps.reshape(21, 2).astype(np.float32)

    # Translate
    wrist = kps[0].copy()
    kps -= wrist

    # Scale
    scale = np.linalg.norm(kps[9])  # wrist→MCP9 after translation
    if scale > 1e-6:
        kps /= scale

    return kps.flatten()


# ── Dataset2 loading ──────────────────────────────────────────────────────────

def get_dataset2_paths() -> List[Tuple[Path, str]]:
    """
    Walk data/dataset2/own_dataset/ and return (image_path, label) pairs.

    WHY: Each subdirectory is named after the ASL letter it contains (A–Z + space).
    Folder-based labelling is the simplest convention and avoids a separate
    CSV/JSON manifest that could go out of sync with the actual files.

    Returns: list of (Path, str) where str is the folder/class name.
    """
    pairs: List[Tuple[Path, str]] = []
    for class_dir in sorted(DATASET2_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                pairs.append((img_path, label))
    return pairs


def encode_labels(labels: List[str]) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Convert a list of string labels to a (N,) int array plus the fitted encoder.

    WHY: scikit-learn classifiers need integer targets. Returning the encoder
    object lets callers decode predictions back to letter strings later.
    """
    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    return y.astype(np.int64), enc


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: str | Path,
) -> None:
    """
    Save a colour-coded confusion matrix as a PNG.

    WHY: A raw accuracy number hides which letters are confused with each other.
    The confusion matrix lets us see whether, e.g., the model confuses M/N (which
    share a similar hand shape) vs random errors — very useful for debugging.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell with its count
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix → {save_path}")
