"""
extract_keypoints_asl_citizen_trimmed.py — Extract MediaPipe keypoints from
ASL Citizen videos, sampling only from the window where hands are detected.

Same fix as extract_keypoints_trimmed.py applied to ASL Citizen:
  Scan 96 frames → find hand window → subsample 32 frames from that window.

Output: data/processed/asl_citizen_keypoints_trimmed/{video_stem}.npy

Run from the project root:
    python -m src.scripts.videos.extract_keypoints_asl_citizen_trimmed
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASL_DIR      = PROJECT_ROOT / "data" / "ASL_Citizen"
VIDEO_DIR    = ASL_DIR / "videos"
SPLITS_DIR   = ASL_DIR / "splits"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed" / "asl_citizen_keypoints_trimmed"
MODELS_DIR   = PROJECT_ROOT / "models"

HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_FRAMES  = 32
NUM_WORKERS = max(1, int((os.cpu_count() or 4) * 0.75))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── Model setup ───────────────────────────────────────────────────────────────

def _download(url, dest):
    log.info("Downloading %s ...", dest.name)
    urllib.request.urlretrieve(url, dest)
    log.info("  saved %d MB", dest.stat().st_size // 1_000_000)


def ensure_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not HAND_MODEL_PATH.exists():
        _download(HAND_MODEL_URL, HAND_MODEL_PATH)
    if not POSE_MODEL_PATH.exists():
        _download(POSE_MODEL_URL, POSE_MODEL_PATH)


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


# ── Keypoint helpers ──────────────────────────────────────────────────────────

def _normalise_hand_3d(lm):
    kps = lm.astype(np.float32)
    kps -= kps[0]
    scale = np.linalg.norm(kps[9])
    if scale > 1e-6:
        kps /= scale
    return kps.flatten()


def _frame_features(hand_res, pose_res):
    left  = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)
    if hand_res.hand_landmarks:
        for lms, cats in zip(hand_res.hand_landmarks, hand_res.handedness):
            arr    = np.array([[l.x, l.y, l.z] for l in lms], dtype=np.float32)
            normed = _normalise_hand_3d(arr)
            if cats[0].category_name == "Left":
                left = normed
            else:
                right = normed
    pose = np.zeros(99, dtype=np.float32)
    if pose_res.pose_landmarks:
        pose = np.array(
            [[l.x, l.y, l.z] for l in pose_res.pose_landmarks[0]], dtype=np.float32
        ).flatten()
    return np.concatenate([left, right, pose])


# ── Core video processing ─────────────────────────────────────────────────────

def _process_video(video_path, hand_det, pose_det):
    """
    Pass 1: sample MAX_FRAMES evenly over the full video, run hand detection only,
            record the actual video frame numbers of the first and last detections.
    Pass 2: sample MAX_FRAMES evenly between those two frame numbers and extract
            full keypoints (hand + pose).
    Falls back to the full video range if no hands found in pass 1.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None, 0, 0

    # ── Pass 1: find the signing window ───────────────────────────────────────
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

    # ── Pass 2: extract keypoints within the signing window ───────────────────
    final_indices = np.linspace(first_frame, last_frame, MAX_FRAMES, dtype=int).tolist()

    frames_kp       = []
    miss_lh, miss_rh = 0, 0

    for vid_idx in final_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            frames_kp.append(np.zeros(225, dtype=np.float32))
            miss_lh += 1
            miss_rh += 1
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        hand_res  = hand_det.detect(mp_img)
        pose_res  = pose_det.detect(mp_img)

        detected = {c[0].category_name for c in hand_res.handedness} if hand_res.handedness else set()
        if "Left"  not in detected: miss_lh += 1
        if "Right" not in detected: miss_rh += 1

        frames_kp.append(_frame_features(hand_res, pose_res))

    cap.release()
    return (np.stack(frames_kp) if frames_kp else None), miss_lh, miss_rh


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(video_stems: list[str], progress_queue) -> dict:
    stats = {"processed": 0, "no_file": 0, "unreadable": 0,
             "miss_lh": 0, "miss_rh": 0, "frames": 0}

    with _make_hand_landmarker() as hand_det, _make_pose_landmarker() as pose_det:
        for stem in video_stems:
            out_path = OUTPUT_DIR / f"{stem}.npy"
            if out_path.exists():
                progress_queue.put(1)
                continue

            vp = VIDEO_DIR / f"{stem}.mp4"
            if not vp.exists():
                stats["no_file"] += 1
                progress_queue.put(1)
                continue

            kps, ml, mr = _process_video(vp, hand_det, pose_det)
            if kps is None:
                stats["unreadable"] += 1
                progress_queue.put(1)
                continue

            np.save(out_path, kps)
            stats["processed"] += 1
            stats["miss_lh"]   += ml
            stats["miss_rh"]   += mr
            stats["frames"]    += kps.shape[0]
            progress_queue.put(1)

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_models()

    all_stems = []
    seen      = set()
    for csv_file in ["train.csv", "val.csv", "test.csv"]:
        df = pd.read_csv(SPLITS_DIR / csv_file)
        for fname in df["Video file"]:
            stem = Path(fname).stem
            if stem not in seen:
                all_stems.append(stem)
                seen.add(stem)

    todo         = [s for s in all_stems if not (OUTPUT_DIR / f"{s}.npy").exists()]
    already_done = len(all_stems) - len(todo)

    log.info("Total videos in index : %d", len(all_stems))
    log.info("Already processed     : %d", already_done)
    log.info("To process this run   : %d", len(todo))
    log.info("Workers               : %d  (MAX_FRAMES=%d, two-pass trimming)",
             NUM_WORKERS, MAX_FRAMES)

    if not todo:
        log.info("Nothing to do.")
        return

    chunks = [todo[i::NUM_WORKERS] for i in range(NUM_WORKERS)]
    totals = {"processed": 0, "no_file": 0, "unreadable": 0,
              "miss_lh": 0, "miss_rh": 0, "frames": 0}

    with Manager() as manager:
        progress_queue = manager.Queue()

        def _drain(pbar, total):
            count = 0
            while count < total:
                progress_queue.get()
                pbar.update(1)
                count += 1

        with tqdm(total=len(todo), desc="Extracting trimmed keypoints") as pbar:
            drain_thread = threading.Thread(
                target=_drain, args=(pbar, len(todo)), daemon=True)
            drain_thread.start()

            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
                futures = {pool.submit(_worker, chunk, progress_queue): i
                           for i, chunk in enumerate(chunks)}
                for fut in as_completed(futures):
                    stats = fut.result()
                    for k in totals:
                        totals[k] += stats.get(k, 0)

            drain_thread.join()

    total_done = already_done + totals["processed"]
    log.info("─" * 60)
    log.info("Processed this run     : %d", totals["processed"])
    log.info("Skipped (no file)      : %d", totals["no_file"])
    log.info("Skipped (unreadable)   : %d", totals["unreadable"])
    log.info("Total .npy files ready : %d", total_done)
    if totals["processed"] > 0:
        log.info("Avg frames/video       : %.1f", totals["frames"] / totals["processed"])
        log.info("Left hand missing      : %.1f%%",
                 100 * totals["miss_lh"] / max(totals["frames"], 1))
        log.info("Right hand missing     : %.1f%%",
                 100 * totals["miss_rh"] / max(totals["frames"], 1))
    log.info("─" * 60)


if __name__ == "__main__":
    main()
