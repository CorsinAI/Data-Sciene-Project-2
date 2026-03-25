import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import models
import torch.nn as nn

# gleiche Parameter wie Training
NUM_FRAMES = 24
IMG_SIZE = 224

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "checkpoints" / "best_custom_wlasl_resnet18.pt"
LABEL_MAP_PATH = PROJECT_ROOT / "data" / "processed" / "custom_wlasl_label_mapping.csv"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame.astype(np.float32) / 255.0
    frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
    return frame


def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = np.linspace(0, total_frames-1, NUM_FRAMES).astype(int)

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        else:
            frame = preprocess_frame(frame)

        frames.append(frame)

    cap.release()

    frames = np.stack(frames)

    return torch.tensor(frames).permute(0,3,1,2).unsqueeze(0)


class VideoClassifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        backbone = models.resnet18(weights=None)

        feature_dim = backbone.fc.in_features

        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):

        b,t,c,h,w = x.shape

        x = x.view(b*t,c,h,w)

        features = self.backbone(x)

        features = features.view(b,t,-1)

        features = features.mean(dim=1)

        features = self.dropout(features)

        return self.classifier(features)


def predict_video(video_path):

    label_map = pd.read_csv(LABEL_MAP_PATH)

    num_classes = len(label_map)

    model = VideoClassifier(num_classes)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    model.to(DEVICE)

    model.eval()

    frames = load_video_frames(video_path).to(DEVICE)

    with torch.no_grad():

        outputs = model(frames)

        pred = outputs.argmax(dim=1).item()

    return label_map.iloc[pred]["gloss"]