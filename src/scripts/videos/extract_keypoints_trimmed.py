"""
extract_keypoints_trimmed.py — Extract MediaPipe keypoints from WLASL videos,
sampling only from the window where hands are actually detected.

Problem with the original script: 32 frames were sampled evenly from the full
video, so ~33% of frames (14.9% leading + 18.0% trailing) contained no hands
because the signer hadn't started or had already finished.

Fix: scan 96 frames from the full video, find the first and last frame where
at least one hand is detected, then subsample 32 frames from that window only.
Falls back to the full range if no hands are detected at all.

Output: data/processed/wlasl_keypoints_trimmed/{video_id}.npy  shape: (T, 225)

Run from the project root:
    python -m src.scripts.videos.extract_keypoints_trimmed
"""

from __future__ import annotations

import json
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
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR     = PROJECT_ROOT / "data" / "dataset 1 videos" / "sign language videos"
WLASL_JSON   = DATA_DIR / "WLASL_v0.3.json"
NSLT_SUBSET  = "nslt_2000"
NSLT_JSON    = DATA_DIR / f"{NSLT_SUBSET}.json"
VIDEO_DIR    = DATA_DIR / "videos"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed" / "wlasl_keypoints_trimmed"
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
MAX_FRAMES   = 32    # frames saved per video
SCAN_FRAMES  = 96    # frames scanned to find the hand window (3× MAX_FRAMES)
NUM_WORKERS  = max(1, (os.cpu_count() or 4) // 2)

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
    1. Sample SCAN_FRAMES evenly from the full video and extract keypoints.
    2. Find the first and last scan frame where at least one hand is detected.
    3. Subsample MAX_FRAMES from within that hand window.
    4. Return the (MAX_FRAMES, 225) array.

    Falls back to the full scan range if no hands are detected.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None, 0, 0

    # ── Pass: extract keypoints for SCAN_FRAMES evenly sampled frames ─────────
    n_scan  = min(total, SCAN_FRAMES)
    indices = np.linspace(0, total - 1, n_scan, dtype=int).tolist()

    scan_kps = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            scan_kps.append(np.zeros(225, dtype=np.float32))
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        hand_res  = hand_det.detect(mp_img)
        pose_res  = pose_det.detect(mp_img)
        scan_kps.append(_frame_features(hand_res, pose_res))

    cap.release()

    scan_kps = np.stack(scan_kps)   # (n_scan, 225)

    # ── Find hand window ───────────────────────────────────────────────────────
    has_hand = np.any(scan_kps[:, 0:126] != 0, axis=1)   # left or right hand present

    if has_hand.any():
        first = int(np.argmax(has_hand))
        last  = int(len(has_hand) - 1 - np.argmax(has_hand[::-1]))
    else:
        # No hands found at all — keep full range so file is still saved
        first, last = 0, len(scan_kps) - 1

    window = scan_kps[first:last + 1]   # (window_len, 225)

    # ── Subsample MAX_FRAMES from the window ──────────────────────────────────
    W = len(window)
    if W <= MAX_FRAMES:
        final = window
    else:
        idx   = np.linspace(0, W - 1, MAX_FRAMES, dtype=int)
        final = window[idx]

    # ── Miss stats on the final frames ────────────────────────────────────────
    miss_lh = int(np.sum(~np.any(final[:, 0:63]  != 0, axis=1)))
    miss_rh = int(np.sum(~np.any(final[:, 63:126] != 0, axis=1)))

    return final.astype(np.float32), miss_lh, miss_rh


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(video_ids: list[str], progress_queue) -> dict:
    stats = {"processed": 0, "no_file": 0, "unreadable": 0,
             "miss_lh": 0, "miss_rh": 0, "frames": 0}

    with _make_hand_landmarker() as hand_det, _make_pose_landmarker() as pose_det:
        for vid_id in video_ids:
            out_path = OUTPUT_DIR / f"{vid_id}.npy"
            if out_path.exists():
                progress_queue.put(1)
                continue

            vp = VIDEO_DIR / f"{vid_id}.mp4"
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


# ── Data index ────────────────────────────────────────────────────────────────

def _build_video_index() -> list[str]:
    with open(WLASL_JSON) as f:
        wlasl = json.load(f)
    with open(NSLT_JSON) as f:
        nslt = json.load(f)

    nslt_ids = set(nslt.keys())
    ids, seen = [], set()
    for entry in wlasl:
        for inst in entry["instances"]:
            vid_id = inst["video_id"]
            if vid_id in nslt_ids and vid_id not in seen:
                ids.append(vid_id)
                seen.add(vid_id)
    return ids


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_models()

    all_ids = _build_video_index()
    todo    = [v for v in all_ids if not (OUTPUT_DIR / f"{v}.npy").exists()]

    already_done = len(all_ids) - len(todo)
    log.info("Total videos in index : %d", len(all_ids))
    log.info("Already processed     : %d", already_done)
    log.info("To process this run   : %d", len(todo))
    log.info("Workers               : %d  (SCAN_FRAMES=%d → MAX_FRAMES=%d)",
             NUM_WORKERS, SCAN_FRAMES, MAX_FRAMES)

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
