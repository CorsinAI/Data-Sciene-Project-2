from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "custom" / "videos"
CUSTOM_METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "custom" / "custom_metadata.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "custom_wlasl"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

for folder in [PROCESSED_DIR, REPORT_DIR, CHECKPOINT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Files
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_custom_wlasl_resnet18.pt"
LABEL_MAP_PATH = PROCESSED_DIR / "custom_wlasl_label_mapping.csv"
SPLIT_METADATA_PATH = PROCESSED_DIR / "custom_wlasl_metadata_with_splits.csv"

HISTORY_PATH = REPORT_DIR / "custom_wlasl_training_history.csv"
CONFUSION_MATRIX_PATH = REPORT_DIR / "custom_wlasl_confusion_matrix.csv"
PER_CLASS_PATH = REPORT_DIR / "custom_wlasl_per_class_performance.csv"
CONFUSION_PAIRS_PATH = REPORT_DIR / "custom_wlasl_confusion_pairs.csv"
MISCLASSIFIED_PATH = REPORT_DIR / "custom_wlasl_misclassified_samples.csv"
ALL_PREDICTIONS_PATH = REPORT_DIR / "custom_wlasl_all_test_predictions.csv"

# Training settings
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# 24 frames per video
NUM_FRAMES = 24
# 224x224 video for ResNet18
IMG_SIZE = 224

BATCH_SIZE = 4
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
SEED = 42

# stop if no improvement after 3 epochs
PATIENCE = 3

TEST_SIZE = 0.15
VAL_SIZE = 0.15

MIN_VIDEOS_PER_CLASS = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalization values
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_metadata_from_csv(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find custom metadata file: {metadata_path}")

    df = pd.read_csv(metadata_path)

    required_columns = {"file_path", "gloss"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Metadata file is missing required columns: {sorted(missing_columns)}")

    df = df.copy()

    df["video_path"] = df["file_path"].astype(str).apply(lambda p: str(Path(p).resolve()))

    df = df[df["video_path"].apply(lambda p: Path(p).suffix.lower() in ALLOWED_EXTENSIONS)].copy()

    df["file_exists"] = df["video_path"].apply(lambda p: Path(p).exists())

    missing_df = df[~df["file_exists"]].copy()
    if not missing_df.empty:
        print("\nWarning: these metadata entries point to missing files and will be skipped:")
        print(missing_df[["gloss", "video_path"]].head(20))

    df = df[df["file_exists"]].copy()
    df = df.drop(columns=["file_exists"])

    if df.empty:
        raise ValueError(f"No valid videos found in metadata file: {metadata_path}")

    return df


def sample_frame_indices(total_frames: int, num_frames: int):
    if total_frames <= 0:
        return [0] * num_frames
    return np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int).tolist()


def preprocess_frame(frame, img_size):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (img_size, img_size))
    frame = frame.astype(np.float32) / 255.0
    frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
    return frame


def load_video_frames(video_path: str, num_frames: int, img_size: int):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        empty = np.zeros((num_frames, img_size, img_size, 3), dtype=np.float32)
        return torch.tensor(empty).permute(0, 3, 1, 2)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sample_frame_indices(total_frames, num_frames)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()

        if not success or frame is None:
            frame = np.zeros((img_size, img_size, 3), dtype=np.float32)
        else:
            frame = preprocess_frame(frame, img_size)

        frames.append(frame)

    cap.release()

    frames = np.stack(frames)
    return torch.tensor(frames).permute(0, 3, 1, 2)


class CustomSignDataset(Dataset):
    def __init__(self, df: pd.DataFrame, num_frames: int, img_size: int):
        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames = load_video_frames(row["video_path"], self.num_frames, self.img_size)
        label = int(row["label"])
        return frames, label, row["video_path"]


class VideoClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        x = x.view(batch_size * time_steps, channels, height, width)

        features = self.backbone(x)
        features = features.view(batch_size, time_steps, -1)
        features = features.mean(dim=1)

        features = self.dropout(features)
        return self.classifier(features)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for frames, labels, _ in tqdm(loader, desc="Training", leave=False):
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * frames.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_paths = []
    all_confidences = []

    for frames, labels, paths in tqdm(loader, desc="Evaluating", leave=False):
        frames = frames.to(device)
        labels = labels.to(device)

        outputs = model(frames)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * frames.size(0)

        probs = torch.softmax(outputs, dim=1)
        confidences, preds = probs.max(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_paths.extend(paths)
        all_confidences.extend(confidences.cpu().numpy())

    loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return loss, acc, all_labels, all_preds, all_paths, all_confidences


def main():
    set_seed(SEED)

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"CUSTOM_VIDEO_DIR: {CUSTOM_VIDEO_DIR}")
    print(f"CUSTOM_METADATA_PATH: {CUSTOM_METADATA_PATH}")

    df = build_metadata_from_csv(CUSTOM_METADATA_PATH)

    class_counts = df["gloss"].value_counts()
    valid_classes = class_counts[class_counts >= MIN_VIDEOS_PER_CLASS].index.tolist()
    df = df[df["gloss"].isin(valid_classes)].copy().reset_index(drop=True)

    if df.empty:
        raise ValueError("No classes left after filtering by minimum videos per class.")

    print("Class counts:")
    print(df["gloss"].value_counts())

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["gloss"])
    num_classes = df["label"].nunique()

    print(f"\nNumber of classes used: {num_classes}")
    print(f"Total videos: {len(df)}")

    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"]
    )

    relative_val_size = VAL_SIZE / (1.0 - TEST_SIZE)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=SEED,
        stratify=train_val_df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    full_split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_split_df.to_csv(SPLIT_METADATA_PATH, index=False)

    print("\nTrain:", len(train_df))
    print("Val:  ", len(val_df))
    print("Test: ", len(test_df))

    print("\nPer-class split table:")
    print(pd.crosstab(full_split_df["gloss"], full_split_df["split"]))

    print("\nRandom baseline:", 1 / num_classes)

    train_dataset = CustomSignDataset(train_df, NUM_FRAMES, IMG_SIZE)
    val_dataset = CustomSignDataset(val_df, NUM_FRAMES, IMG_SIZE)
    test_dataset = CustomSignDataset(test_df, NUM_FRAMES, IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = VideoClassifier(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    print("\nModel created on:", DEVICE)

    best_val_acc = 0.0
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved best model to {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))

    test_loss, test_acc, y_true, y_pred, test_paths, test_confidences = evaluate(
        model, test_loader, criterion, DEVICE
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))

    class_names = label_encoder.classes_
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    print("\nConfusion Matrix:")
    print(cm_df)

    per_class_stats = []
    for i, class_name in enumerate(class_names):
        total_true = cm[i].sum()
        correct = cm[i, i]
        recall = correct / total_true if total_true > 0 else 0.0

        total_pred = cm[:, i].sum()
        precision = correct / total_pred if total_pred > 0 else 0.0

        per_class_stats.append({
            "gloss": class_name,
            "support": int(total_true),
            "correct": int(correct),
            "errors": int(total_true - correct),
            "recall": recall,
            "precision": precision
        })

    per_class_df = pd.DataFrame(per_class_stats).sort_values(
        by=["recall", "precision", "errors"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    print("\nPer-class performance (worst classes first):")
    print(per_class_df)

    confusion_pairs = []
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                confusion_pairs.append({
                    "true_gloss": class_names[true_idx],
                    "predicted_gloss": class_names[pred_idx],
                    "count": int(cm[true_idx, pred_idx])
                })

    if confusion_pairs:
        confusion_pairs_df = pd.DataFrame(confusion_pairs).sort_values(
            by="count", ascending=False
        ).reset_index(drop=True)
    else:
        confusion_pairs_df = pd.DataFrame(columns=["true_gloss", "predicted_gloss", "count"])

    print("\nMost common confusions:")
    print(confusion_pairs_df.head(20))

    misclassified_rows = []
    for path, true_label, pred_label, confidence in zip(test_paths, y_true, y_pred, test_confidences):
        if true_label != pred_label:
            misclassified_rows.append({
                "video_path": path,
                "true_gloss": class_names[true_label],
                "predicted_gloss": class_names[pred_label],
                "confidence": float(confidence)
            })

    if misclassified_rows:
        misclassified_df = pd.DataFrame(misclassified_rows).sort_values(
            by="confidence", ascending=False
        ).reset_index(drop=True)
    else:
        misclassified_df = pd.DataFrame(columns=["video_path", "true_gloss", "predicted_gloss", "confidence"])

    print("\nSome misclassified samples:")
    print(misclassified_df.head(20))

    all_predictions_rows = []
    for path, true_label, pred_label, confidence in zip(test_paths, y_true, y_pred, test_confidences):
        all_predictions_rows.append({
            "video_path": path,
            "true_gloss": class_names[true_label],
            "predicted_gloss": class_names[pred_label],
            "confidence": float(confidence),
            "correct": int(true_label == pred_label)
        })

    all_predictions_df = pd.DataFrame(all_predictions_rows)

    print("\nWorst classes by recall:")
    print(per_class_df[["gloss", "recall", "precision", "support", "errors"]])

    label_map_df = pd.DataFrame({
        "gloss": label_encoder.classes_,
        "label": range(len(label_encoder.classes_))
    })
    label_map_df.to_csv(LABEL_MAP_PATH, index=False)

    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_PATH, index=False)

    cm_df.to_csv(CONFUSION_MATRIX_PATH)
    per_class_df.to_csv(PER_CLASS_PATH, index=False)
    confusion_pairs_df.to_csv(CONFUSION_PAIRS_PATH, index=False)
    misclassified_df.to_csv(MISCLASSIFIED_PATH, index=False)
    all_predictions_df.to_csv(ALL_PREDICTIONS_PATH, index=False)

    print(f"\nSaved metadata to {SPLIT_METADATA_PATH}")
    print(f"Saved label mapping to {LABEL_MAP_PATH}")
    print(f"Saved training history to {HISTORY_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")
    print(f"Saved per-class performance to {PER_CLASS_PATH}")
    print(f"Saved confusion pairs to {CONFUSION_PAIRS_PATH}")
    print(f"Saved misclassified samples to {MISCLASSIFIED_PATH}")
    print(f"Saved all test predictions to {ALL_PREDICTIONS_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()