"""
classify_and_compare.py — Extract hand keypoints from dataset2 and compare two classifiers.

═══════════════════════════════════════════════════════════════════════════════════════
WHY THIS SCRIPT EXISTS
═══════════════════════════════════════════════════════════════════════════════════════
We want to answer one central question:

    "Is a keypoint extractor trained on our own dataset (dataset1) good enough to
     classify ASL letters, and how does it compare to Google's MediaPipe — a
     production-grade, hand-specialist model?"

To answer this fairly, BOTH approaches go through the exact same downstream
classifier (a 2-hidden-layer MLP) trained on the same 80/20 split with the same
random seed.  Any difference in final accuracy therefore comes from the quality of
the keypoints themselves, not from classifier differences.

═══════════════════════════════════════════════════════════════════════════════════════
PIPELINE OVERVIEW
═══════════════════════════════════════════════════════════════════════════════════════

  dataset2 images
      │
      ├── [Custom path]  → KeypointRegressor (trained by train_keypoint_model.py)
      │                  → normalise_keypoints()
      │                  → MLPClassifier  →  accuracy / F1
      │
      └── [MediaPipe path] → mp.solutions.hands.Hands()
                           → normalise_keypoints()
                           → MLPClassifier  →  accuracy / F1
                                                    │
                                          side-by-side comparison table
                                          confusion matrix PNGs
                                          comparison_report.csv

═══════════════════════════════════════════════════════════════════════════════════════
OUTPUTS
═══════════════════════════════════════════════════════════════════════════════════════
  checkpoints/asl_classifier_custom.pkl
  checkpoints/asl_classifier_mediapipe.pkl
  src/scripts/pictures/results/confusion_custom.png
  src/scripts/pictures/results/confusion_mediapipe.png
  src/scripts/pictures/results/comparison_report.csv

USAGE
═══════════════════════════════════════════════════════════════════════════════════════
  python -m src.scripts.pictures.classify_and_compare

  Requires: checkpoints/hand_keypoint_regressor.pt (run train_keypoint_model.py first)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from src.scripts.pictures.utils import (
    encode_labels,
    get_dataset2_paths,
    image_to_tensor,
    load_image_rgb,
    normalise_keypoints,
    plot_confusion_matrix,
)
from src.scripts.pictures.train_keypoint_model import KeypointRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR  = PROJECT_ROOT / "checkpoints"
REGRESSOR_PATH  = CHECKPOINT_DIR / "hand_keypoint_regressor.pt"
RESULTS_DIR     = Path(__file__).resolve().parent / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42


# ═════════════════════════════════════════════════════════════════════════════
# 1. KEYPOINT EXTRACTION — CUSTOM MODEL
# ═════════════════════════════════════════════════════════════════════════════

def extract_custom(image_paths: list) -> np.ndarray:
    """
    Run all images through the trained KeypointRegressor and return raw predictions.

    WHY WE BATCH:
    GPU inference is dramatically faster with batches than single-image calls.
    For CPU this matters less but still avoids Python-loop overhead from loading
    the model 11 000 times.

    WHY NO NORMALISATION HERE:
    normalise_keypoints() is called after this function so that the same
    normalisation code path is used for both Custom and MediaPipe outputs —
    guaranteeing they are on the same scale when the MLP sees them.

    Returns: np.ndarray shape (N, 42) — raw regressor output in [0, 1] range
             (coordinates normalised to image W/H during training).
    """
    if not REGRESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Custom keypoint regressor not found at {REGRESSOR_PATH}.\n"
            "Run train_keypoint_model.py first."
        )

    print("\n[Custom] Loading keypoint regressor…")
    model = KeypointRegressor().to(DEVICE)
    model.load_state_dict(torch.load(REGRESSOR_PATH, map_location=DEVICE))
    model.eval()

    all_kps = []
    batch_size = 64

    print("[Custom] Extracting keypoints…")
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i : i + batch_size]
        tensors = []
        for p in batch_paths:
            img = load_image_rgb(p)
            tensors.append(image_to_tensor(img))
        batch = torch.stack(tensors).to(DEVICE)   # (B, 3, 224, 224)

        with torch.no_grad():
            preds = model(batch).cpu().numpy()     # (B, 42)
        all_kps.append(preds)

    return np.concatenate(all_kps, axis=0)         # (N, 42)


# ═════════════════════════════════════════════════════════════════════════════
# 2. KEYPOINT EXTRACTION — MEDIAPIPE
# ═════════════════════════════════════════════════════════════════════════════

def extract_mediapipe(image_paths: list) -> tuple[np.ndarray, int]:
    """
    Run all images through MediaPipe Hands and return landmark coordinates.

    WHY static_image_mode=True:
    In video mode MediaPipe tracks the hand across frames for speed.  For still
    images there are no previous frames, so we must use static_image_mode which
    runs the full detection on every image.

    WHY min_detection_confidence=0.5:
    A lenient threshold keeps the failure rate low.  Failed detections return a
    zero-vector (noted in the comparison report).  This is preferable to dropping
    the sample entirely because it keeps the two feature matrices aligned with
    the label array — the failed images are simply harder cases for MediaPipe.

    WHY take the first detected hand only:
    Our dataset2 images contain exactly one hand per image (self-captured signs).
    If MediaPipe somehow detects a second hand (artefact or mirror reflection) we
    want the dominant one, which MediaPipe returns first.

    Returns:
        features  — (N, 42) float32 array of (x, y) landmarks in [0, 1]
        n_failed  — number of images where no hand was detected
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    print("\n[MediaPipe] Extracting keypoints…")

    # Download the hand landmarker model if not already cached
    model_path = Path(__file__).resolve().parents[3] / "checkpoints" / "hand_landmarker.task"
    if not model_path.exists():
        import urllib.request
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print("  Downloading hand_landmarker.task model…")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            model_path,
        )

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_kps  = []
    n_failed = 0

    with mp_vision.HandLandmarker.create_from_options(options) as detector:
        for p in tqdm(image_paths):
            img = load_image_rgb(p)  # RGB uint8 (224, 224, 3)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]  # first hand, list of 21 NormalizedLandmark
                xy = np.array([[l.x, l.y] for l in lm], dtype=np.float32)  # (21, 2)
                all_kps.append(xy.flatten())
            else:
                # No hand detected — use zero-vector so the index stays aligned
                all_kps.append(np.zeros(42, dtype=np.float32))
                n_failed += 1

    return np.array(all_kps, dtype=np.float32), n_failed


# ═════════════════════════════════════════════════════════════════════════════
# 3. NORMALISE KEYPOINTS
# ═════════════════════════════════════════════════════════════════════════════

def normalise_batch(raw_features: np.ndarray) -> np.ndarray:
    """
    Apply normalise_keypoints() to every row in the feature matrix.

    WHY: The regressor and MediaPipe both output absolute [0,1] coordinates that
    encode hand position within the frame.  Two identical signs at different
    positions in the frame would produce different feature vectors without
    normalisation, making the classifier's job much harder.
    """
    return np.array([normalise_keypoints(row) for row in raw_features])


# ═════════════════════════════════════════════════════════════════════════════
# 4. CLASSIFIER
# ═════════════════════════════════════════════════════════════════════════════

def build_mlp() -> MLPClassifier:
    """
    Build the shared MLP classifier configuration.

    WHY MLPClassifier (sklearn) rather than a PyTorch model:
    • We already have a PyTorch model for feature extraction; a second one would
      add boilerplate without benefit for what is essentially a small tabular task
      (11 k samples × 42 features).
    • sklearn's MLPClassifier provides early stopping out of the box and is
      trivially serialisable with joblib — important for later integration into
      the Streamlit demo.

    WHY 256 → 128 architecture:
    42 input features is small.  Two moderate hidden layers give the model enough
    capacity to learn non-linear decision boundaries between similar signs
    (e.g. M/N/T which differ in subtle finger positions) without overfitting.
    """
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=SEED,
        verbose=False,
    )


def train_and_eval(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    name: str,
) -> dict:
    """
    Train an MLP on 80% of the data, evaluate on 20%, save the model and confusion matrix.

    WHY StratifiedShuffleSplit instead of train_test_split:
    StratifiedShuffleSplit guarantees that each class is represented proportionally
    in both train and test sets.  Without stratification a class with few images
    could end up entirely in one split, giving a misleading accuracy number.

    Returns a metrics dict for the comparison report.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(sss.split(features, labels))

    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx],   labels[test_idx]

    print(f"\n[{name}] Training MLP…  ({len(X_train)} train / {len(X_test)} test)")
    clf = build_mlp()
    clf.fit(X_train, y_train)

    y_pred    = clf.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    report    = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    macro_f1  = report["macro avg"]["f1-score"]
    cm        = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy : {accuracy*100:.2f}%")
    print(f"  Macro F1 : {macro_f1:.4f}")

    # Confusion matrix
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = RESULTS_DIR / f"confusion_{name.lower()}.png"
    plot_confusion_matrix(cm, class_names, title=f"Confusion Matrix — {name}", save_path=cm_path)

    # Save per-class report
    per_class = pd.DataFrame(report).T.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    per_class.to_csv(RESULTS_DIR / f"per_class_{name.lower()}.csv")

    # Save classifier
    clf_path = CHECKPOINT_DIR / f"asl_classifier_{name.lower()}.pkl"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, clf_path)
    print(f"  Saved classifier → {clf_path}")

    return {
        "name":          name,
        "accuracy":      accuracy,
        "macro_f1":      macro_f1,
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. COMPARISON REPORT
# ═════════════════════════════════════════════════════════════════════════════

def print_comparison(results: list[dict], n_failed_mediapipe: int, n_total: int):
    """
    Print a side-by-side summary and save it as a CSV.

    WHY include failed-detection count:
    A high failure rate would inflate MediaPipe's apparent accuracy if we simply
    dropped failed images (they'd no longer be in the test set).  By keeping
    zero-vectors for failed images and reporting the count, the reader can see
    the full picture: MediaPipe might have a higher accuracy on the images it
    *did* detect but a meaningful miss rate on others.
    """
    print("\n" + "═" * 55)
    print("  COMPARISON REPORT")
    print("═" * 55)
    print(f"  {'Metric':<28} {'Custom':>10}  {'MediaPipe':>10}")
    print("─" * 55)

    c = {r["name"]: r for r in results}
    cu = c.get("Custom", {})
    mp = c.get("MediaPipe", {})

    def row(label, key, fmt="{:.4f}"):
        cv = cu.get(key, float("nan"))
        mv = mp.get(key, float("nan"))
        print(f"  {label:<28} {fmt.format(cv):>10}  {fmt.format(mv):>10}")

    row("Overall Accuracy",  "accuracy",  "{:.2%}")
    row("Macro F1",          "macro_f1",  "{:.4f}")
    print(f"  {'Failed detections':<28} {'0':>10}  {n_failed_mediapipe:>10} ({n_failed_mediapipe/n_total:.1%})")
    print("═" * 55)

    # Save CSV
    df = pd.DataFrame(results)
    df["mediapipe_failed_detections"] = [0, n_failed_mediapipe]
    df["mediapipe_failed_pct"]        = [0.0, n_failed_mediapipe / n_total]
    csv_path = RESULTS_DIR / "comparison_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Full report saved → {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load dataset2 paths + labels ──────────────────────────────────────
    print("Loading dataset2 paths…")
    pairs = get_dataset2_paths()
    image_paths = [str(p) for p, _ in pairs]
    raw_labels  = [lbl for _, lbl in pairs]

    labels_int, encoder = encode_labels(raw_labels)
    class_names = list(encoder.classes_)

    print(f"  Total images : {len(pairs)}")
    print(f"  Classes      : {len(class_names)}  →  {class_names}")

    # ── Custom pipeline ────────────────────────────────────────────────────
    raw_custom    = extract_custom(image_paths)
    norm_custom   = normalise_batch(raw_custom)
    result_custom = train_and_eval(norm_custom, labels_int, class_names, name="Custom")

    # ── MediaPipe pipeline ─────────────────────────────────────────────────
    raw_mediapipe, n_failed = extract_mediapipe(image_paths)
    norm_mediapipe          = normalise_batch(raw_mediapipe)
    result_mediapipe        = train_and_eval(norm_mediapipe, labels_int, class_names, name="MediaPipe")

    # ── Comparison report ──────────────────────────────────────────────────
    print_comparison([result_custom, result_mediapipe], n_failed, len(pairs))


if __name__ == "__main__":
    main()
