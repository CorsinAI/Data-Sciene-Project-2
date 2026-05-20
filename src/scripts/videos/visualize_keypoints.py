"""
visualize_keypoints.py — Extract the 32 trimmed frames from a video and draw
all keypoints used by the transformer models:
  - Left hand  : 21 landmarks  (red)
  - Right hand : 21 landmarks  (green)
  - Pose       : 33 landmarks  (blue)
Total: 225 features per frame (63 + 63 + 99).

Output: outputs/keypoint_viz/{video_stem}.gif  (opens in any browser or image viewer)

Usage (from project root):
    python -m src.scripts.videos.visualize_keypoints <video_stem_or_path>

Examples:
    python -m src.scripts.videos.visualize_keypoints 29410473817374494-CUTE
    python -m src.scripts.videos.visualize_keypoints "data/ASL_Citizen/videos/29410473817374494-CUTE.mp4"
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
VIDEO_DIR    = PROJECT_ROOT / "data" / "ASL_Citizen" / "videos"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "keypoint_viz"
MODELS_DIR   = PROJECT_ROOT / "models"

HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

MAX_FRAMES        = 32
DOT_RADIUS        = 5
COLOR_LEFT_HAND   = (0,   0,   255)  # red   (BGR)
COLOR_RIGHT_HAND  = (0,   255, 0  )  # green (BGR)
COLOR_POSE        = (255, 0,   0  )  # blue  (BGR)
MS_PER_FRAME      = 250              # 250 ms per frame = 4 fps


# ── Model ─────────────────────────────────────────────────────────────────────

def ensure_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not HAND_MODEL_PATH.exists():
        print("Downloading hand model...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    if not POSE_MODEL_PATH.exists():
        print("Downloading pose model...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)


def _make_hand_landmarker():
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def _make_pose_landmarker():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


# ── Two-pass frame selection ──────────────────────────────────────────────────

def find_signing_window(cap, total, hand_det):
    """Pass 1: scan MAX_FRAMES over the full video, return first/last video frame
    numbers where a hand was detected. Falls back to full range if none found."""
    scan_indices = np.linspace(0, total - 1, MAX_FRAMES, dtype=int).tolist()
    first_frame  = scan_indices[0]
    last_frame   = scan_indices[-1]
    found_first  = False

    for vid_idx in scan_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if hand_det.detect(mp_img).hand_landmarks:
            if not found_first:
                first_frame = vid_idx
                found_first = True
            last_frame = vid_idx

    return first_frame, last_frame, found_first


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_keypoints(frame_bgr, hand_res, pose_res):
    """Draw all keypoints used by the transformer models.
    Left hand = red, right hand = green, pose = blue.
    """
    h, w = frame_bgr.shape[:2]
    out  = frame_bgr.copy()

    # Hand landmarks — colour by chirality
    for i, hand_landmarks in enumerate(hand_res.hand_landmarks):
        label = hand_res.handedness[i][0].category_name  # "Left" or "Right"
        color = COLOR_LEFT_HAND if label == "Left" else COLOR_RIGHT_HAND
        for lm in hand_landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(out, (px, py), DOT_RADIUS, color, thickness=-1)

    # Pose landmarks — blue
    if pose_res.pose_landmarks:
        for lm in pose_res.pose_landmarks[0]:
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(out, (px, py), DOT_RADIUS, COLOR_POSE, thickness=-1)

    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.scripts.videos.visualize_keypoints <video_stem_or_path>")
        sys.exit(1)

    arg        = sys.argv[1]
    video_path = Path(arg)
    if not video_path.exists():
        video_path = VIDEO_DIR / (arg if arg.endswith(".mp4") else f"{arg}.mp4")
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    stem     = video_path.stem
    out_path = OUTPUT_DIR / f"{stem}.gif"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ensure_models()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video      : {video_path.name}")
    print(f"Resolution : {w}x{h}  FPS: {fps:.1f}  Frames: {total}")
    print(f"Keypoints  : left hand (red) + right hand (green) + pose (blue)")

    pil_frames = []

    with _make_hand_landmarker() as hand_det, _make_pose_landmarker() as pose_det:
        # Pass 1 — find signing window using hand detection
        first_frame, last_frame, found = find_signing_window(cap, total, hand_det)
        if found:
            print(f"Signing window : frame {first_frame} → {last_frame} "
                  f"({last_frame - first_frame + 1} frames)")
        else:
            print("No hands detected in scan — using full video range")

        # Pass 2 — extract, annotate, collect frames
        final_indices = np.linspace(first_frame, last_frame, MAX_FRAMES, dtype=int).tolist()
        no_hands = 0

        for i, vid_idx in enumerate(final_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"  [{i:02d}] frame {vid_idx}: could not read")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            hand_res  = hand_det.detect(mp_img)
            pose_res  = pose_det.detect(mp_img)

            annotated_bgr = draw_keypoints(frame, hand_res, pose_res)

            n_hands = len(hand_res.hand_landmarks)
            n_pose  = len(pose_res.pose_landmarks) if pose_res.pose_landmarks else 0
            label   = f"frame {vid_idx}  hands: {n_hands}  pose: {n_pose}"
            cv2.putText(annotated_bgr, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Convert BGR → RGB for PIL
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(annotated_rgb))

            if n_hands == 0:
                no_hands += 1

    cap.release()

    if not pil_frames:
        print("No frames collected — nothing saved.")
        sys.exit(1)

    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=MS_PER_FRAME,
        loop=0,           # loop forever
    )

    print(f"\nSaved {len(pil_frames)}-frame GIF → {out_path}")
    print(f"Frames with no hands detected : {no_hands}/{len(pil_frames)}")
    print("Colours: red = left hand, green = right hand, blue = pose")
    print("Open the .gif in any browser (drag and drop) or Windows Photos.")


if __name__ == "__main__":
    main()
