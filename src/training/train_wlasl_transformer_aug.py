"""
train_wlasl_transformer_aug.py — Transformer encoder + on-the-fly keypoint augmentation.

Combines the architecture from train_wlasl_transformer.py with the augmentations
from train_wlasl_keypoints_aug.py:
  - Speed jitter   : resample sequence at a random rate (0.8x–1.2x)
  - Gaussian noise : small additive noise on all coordinates
  - Frame dropout  : randomly zero out individual frames

Run from the project root:
    python -m src.training.train_wlasl_transformer_aug
    python -m src.training.train_wlasl_transformer_aug --eval-only
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "dataset 1 videos" / "sign language videos"
NSLT_SUBSET  = "nslt_1000"  # change to nslt_100, nslt_300, nslt_2000 to scale up

WLASL_JSON   = DATA_DIR / "WLASL_v0.3.json"
NSLT_JSON    = DATA_DIR / f"{NSLT_SUBSET}.json"
KEYPOINT_DIR = PROJECT_ROOT / "data" / "processed" / "wlasl_keypoints_nslt_2000"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REPORT_DIR   = PROJECT_ROOT / "outputs" / "reports" / f"wlasl_keypoints_{NSLT_SUBSET}_transformer_aug"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

for d in [CHECKPOINT_DIR, REPORT_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH    = CHECKPOINT_DIR / f"wlasl_transformer_{NSLT_SUBSET}_aug.pt"
LABEL_MAP_PATH     = PROCESSED_DIR / f"wlasl_keypoint_label_mapping_{NSLT_SUBSET}.csv"
HISTORY_PATH       = PROCESSED_DIR / f"wlasl_transformer_train_history_{NSLT_SUBSET}_aug.csv"
CONFUSION_PATH     = REPORT_DIR / "wlasl_transformer_confusion_matrix.csv"
PER_CLASS_PATH     = REPORT_DIR / "wlasl_transformer_per_class.csv"
MISCLASSIFIED_PATH = REPORT_DIR / "wlasl_transformer_misclassified.csv"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
SEQ_LEN      = 32
INPUT_DIM    = 225
D_MODEL      = 256
NHEAD        = 8
NUM_LAYERS   = 4
DIM_FF       = 512
DROPOUT      = 0.3
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 100
PATIENCE     = 20
SEED         = 42
NUM_WORKERS  = 0

# ── Augmentation parameters ───────────────────────────────────────────────────
SPEED_MIN    = 0.7   # ↑ wider speed range
SPEED_MAX    = 1.3
NOISE_STD    = 0.02  # ↑ more noise
FRAME_DROP_P = 0.15  # ↑ more frame dropout

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Data preparation ──────────────────────────────────────────────────────────

def load_split_records() -> list[tuple[str, str, str]]:
    with open(WLASL_JSON) as f:
        wlasl = json.load(f)
    with open(NSLT_JSON) as f:
        nslt = json.load(f)

    records: list[tuple[str, str, str]] = []
    for entry in wlasl:
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            vid_id = inst["video_id"]
            if vid_id not in nslt:
                continue
            if not (KEYPOINT_DIR / f"{vid_id}.npy").exists():
                continue
            records.append((vid_id, gloss, nslt[vid_id]["subset"]))
    return records


def build_label_map(records: list[tuple[str, str, str]]) -> dict[str, int]:
    glosses = sorted({gloss for _, gloss, _ in records})
    return {g: i for i, g in enumerate(glosses)}


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_sequence(seq: np.ndarray) -> np.ndarray:
    T = seq.shape[0]

    # Speed jitter
    rate    = np.random.uniform(SPEED_MIN, SPEED_MAX)
    new_len = max(1, int(round(T * rate)))
    indices = np.linspace(0, T - 1, new_len).astype(int)
    seq     = seq[indices]
    indices = np.linspace(0, len(seq) - 1, T).astype(int)
    seq     = seq[indices]

    # Gaussian noise
    seq = seq + np.random.normal(0, NOISE_STD, seq.shape).astype(np.float32)

    # Frame dropout
    drop_mask = np.random.rand(T) < FRAME_DROP_P
    seq[drop_mask] = 0.0

    return seq


# ── Dataset ───────────────────────────────────────────────────────────────────

class WLASLKeypointDataset(Dataset):
    def __init__(self, records, label_map, seq_len=SEQ_LEN, augment=False):
        self.records   = records
        self.label_map = label_map
        self.seq_len   = seq_len
        self.augment   = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        vid_id, gloss, _ = self.records[idx]
        label = self.label_map[gloss]
        raw   = np.load(KEYPOINT_DIR / f"{vid_id}.npy")

        if self.augment:
            raw = augment_sequence(raw)

        T = raw.shape[0]
        if T >= self.seq_len:
            indices = np.linspace(0, T - 1, self.seq_len, dtype=int)
            seq     = raw[indices]
            mask    = np.ones(self.seq_len, dtype=bool)
        else:
            pad  = np.zeros((self.seq_len - T, raw.shape[1]), dtype=np.float32)
            seq  = np.concatenate([raw, pad], axis=0)
            mask = np.zeros(self.seq_len, dtype=bool)
            mask[:T] = True

        return (torch.from_numpy(seq.astype(np.float32)),
                label,
                torch.from_numpy(mask))


# ── Positional encoding ───────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── Model ─────────────────────────────────────────────────────────────────────

class SignLanguageTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_ff,
                 num_classes, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask):
        x            = self.pos_enc(self.input_proj(x))
        padding_mask = ~mask
        x            = self.encoder(x, src_key_padding_mask=padding_mask)
        real_mask    = mask.unsqueeze(-1).float()
        pooled       = (x * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1)
        return self.classifier(self.dropout(pooled))


# ── Helpers ───────────────────────────────────────────────────────────────────

def topk_accuracy(outputs, labels, k=5):
    with torch.no_grad():
        topk    = outputs.topk(min(k, outputs.size(1)), dim=1).indices
        correct = topk.eq(labels.unsqueeze(1).expand_as(topk)).any(dim=1)
        return correct.float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for seqs, labels, masks in tqdm(loader, desc="Train", leave=False):
        seqs, labels, masks = seqs.to(device), labels.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(seqs, masks)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    top5_sum, n_batches = 0.0, 0
    for seqs, labels, masks in tqdm(loader, desc="Eval", leave=False):
        seqs, labels, masks = seqs.to(device), labels.to(device), masks.to(device)
        logits = model(seqs, masks)
        total_loss += criterion(logits, labels).item() * seqs.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        top5_sum  += topk_accuracy(logits, labels, k=5)
        n_batches += 1
    top1 = accuracy_score(all_labels, all_preds)
    top5 = top5_sum / n_batches if n_batches > 0 else 0.0
    return total_loss / len(loader.dataset), top1, top5, all_labels, all_preds, []


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    set_seed()
    print(f"Device: {DEVICE}")
    print(f"Model : TransformerEncoder + augmentation  d_model={D_MODEL}  nhead={NHEAD}  "
          f"layers={NUM_LAYERS}  dim_ff={DIM_FF}  dropout={DROPOUT}")
    print(f"Aug   : speed [{SPEED_MIN}x–{SPEED_MAX}x]  noise std={NOISE_STD}  frame drop p={FRAME_DROP_P}")

    records = load_split_records()
    print(f"Total records: {len(records)}")
    if not records:
        raise RuntimeError("No .npy files found. Run extract_keypoints.py first.")

    label_map   = build_label_map(records)
    num_classes = len(label_map)
    print(f"Classes: {num_classes}  Random baseline: {1/num_classes:.4f}")

    pd.DataFrame(sorted(label_map.items(), key=lambda x: x[1]),
                 columns=["gloss", "label"]).to_csv(LABEL_MAP_PATH, index=False)

    train_recs = [(v, g, s) for v, g, s in records if s == "train"]
    val_recs   = [(v, g, s) for v, g, s in records if s == "val"]
    test_recs  = [(v, g, s) for v, g, s in records if s == "test"]
    print(f"Train: {len(train_recs)}  Val: {len(val_recs)}  Test: {len(test_recs)}")

    train_loader = DataLoader(WLASLKeypointDataset(train_recs, label_map, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    val_loader   = DataLoader(WLASLKeypointDataset(val_recs, label_map, augment=False),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(WLASLKeypointDataset(test_recs, label_map, augment=False),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = SignLanguageTransformer(
        input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD,
        num_layers=NUM_LAYERS, dim_ff=DIM_FF,
        num_classes=num_classes, dropout=DROPOUT,
    ).to(DEVICE)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5, verbose=True)

    if args.eval_only:
        print("--eval-only: skipping training.")
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(f"No checkpoint at {BEST_MODEL_PATH}.")
    else:
        best_val_top1, epochs_no_improve = 0.0, 0
        history = {"epoch": [], "train_loss": [], "train_acc": [],
                   "val_loss": [], "val_top1": [], "val_top5": []}

        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_top1, val_top5, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step(val_top1)

            for k, v in zip(history, [epoch, train_loss, train_acc, val_loss, val_top1, val_top5]):
                history[k].append(v)

            print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.4f}\n"
                  f"  Val    loss={val_loss:.4f}  top1={val_top1:.4f}  top5={val_top5:.4f}")

            if val_top1 > best_val_top1:
                best_val_top1, epochs_no_improve = val_top1, 0
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"  * New best val top-1={best_val_top1:.4f}  saved checkpoint")
            else:
                epochs_no_improve += 1
                print(f"  No improvement ({epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping after {epoch} epochs.")
                break

        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
        print(f"Saved training history → {HISTORY_PATH}")

    print("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))

    test_loss, test_top1, test_top5, y_true, y_pred, _ = evaluate(
        model, test_loader, criterion, DEVICE)
    print(f"\nTest  loss={test_loss:.4f}  top-1={test_top1:.4f}  top-5={test_top5:.4f}")

    idx_to_gloss   = {v: k for k, v in label_map.items()}
    class_names    = [idx_to_gloss[i] for i in range(num_classes)]
    present_labels = sorted(set(y_true) | set(y_pred))
    present_names  = [idx_to_gloss[i] for i in present_labels]

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=present_labels,
                                 target_names=present_names, zero_division=0))

    cm    = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(CONFUSION_PATH)

    per_class = []
    for i, name in enumerate(class_names):
        total_true = cm[i].sum()
        correct    = cm[i, i]
        total_pred = cm[:, i].sum()
        per_class.append({"gloss": name, "support": int(total_true),
                           "correct": int(correct),
                           "recall": correct / total_true if total_true > 0 else 0.0,
                           "precision": correct / total_pred if total_pred > 0 else 0.0})
    pd.DataFrame(per_class).sort_values("recall").reset_index(drop=True).to_csv(PER_CLASS_PATH, index=False)

    misclassified = [
        {"video_id": test_recs[i][0], "true_gloss": class_names[yt],
         "predicted_gloss": class_names[yp]}
        for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt != yp
    ]
    pd.DataFrame(misclassified).to_csv(MISCLASSIFIED_PATH, index=False)

    print(f"\nSaved reports → {REPORT_DIR}")
    print("\nDone.")
    if not args.eval_only:
        print(f"Best val top-1 : {best_val_top1:.4f}")
    print(f"Test  top-1    : {test_top1:.4f}")
    print(f"Test  top-5    : {test_top5:.4f}")


if __name__ == "__main__":
    main()
