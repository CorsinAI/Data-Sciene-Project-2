import os
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
# 1. PATHS + CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_WLASL_DIR = PROJECT_ROOT / "data" / "raw" / "WLASL"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

JSON_PATH = RAW_WLASL_DIR / "WLASL_v0.3.json"

MIN_FRAMES = 8
MIN_SAMPLES_PER_GLOSS = 5

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. INSPECT FOLDERS
# =========================================================
for item in os.listdir(RAW_WLASL_DIR):
    print(item)

for current_root, dirs, files in os.walk(RAW_WLASL_DIR):
    print("FOLDER:", current_root)
    print("  Subfolders:", dirs[:5])
    print("  Files:", files[:5])
    print()

# =========================================================
# 3. LOAD JSON
# =========================================================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(type(data))
print("Number of top-level entries:", len(data))
print(data[0].keys())
print(data[0])

# =========================================================
# 4. FLATTEN JSON TO DATAFRAME
# =========================================================
rows = []

for entry in data:
    gloss = entry.get("gloss")
    gloss_id = entry.get("gloss_id")
    instances = entry.get("instances", [])

    for inst in instances:
        row = {
            "gloss": gloss,
            "gloss_id": gloss_id,
            "video_id": inst.get("video_id"),
            "split": inst.get("split"),
            "signer_id": inst.get("signer_id"),
            "instance_id": inst.get("instance_id"),
            "source": inst.get("source"),
            "variation_id": inst.get("variation_id"),
        }
        rows.append(row)

df_wlasl = pd.DataFrame(rows)

print(df_wlasl.head())
print(df_wlasl.shape)
print(df_wlasl.columns.tolist())
print("-------------")

# =========================================================
# 5. BASIC STATS
# =========================================================
print("Total video samples:", len(df_wlasl))
print("Total glosses:", df_wlasl["gloss"].nunique())
print("Total signers:", df_wlasl["signer_id"].nunique())
print()
print(df_wlasl["split"].value_counts(dropna=False))

print("-------------")

gloss_counts = df_wlasl["gloss"].value_counts()

print(gloss_counts.head(20))
print("Min samples per gloss:", gloss_counts.min())
print("Max samples per gloss:", gloss_counts.max())
print("Mean samples per gloss:", gloss_counts.mean())
print("Median samples per gloss:", gloss_counts.median())

print("-------------")

# =========================================================
# 6. PLOT TOP CLASSES
# =========================================================
top_n = 30
gloss_counts.head(top_n).plot(kind="bar", figsize=(12, 5))
plt.title(f"Top {top_n} glosses by number of videos")
plt.xlabel("Gloss")
plt.ylabel("Number of videos")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("-------------")

# =========================================================
# 7. MATCH METADATA TO LOCAL VIDEO FILES
# =========================================================
video_files = []
for current_root, dirs, files in os.walk(RAW_WLASL_DIR):
    for file_name in files:
        if file_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_files.append(os.path.join(current_root, file_name))

print("Number of video files found:", len(video_files))
print(video_files[:5])

print("-------------")

video_lookup = {}
for vp in video_files:
    stem = Path(vp).stem
    video_lookup[stem] = vp

df_wlasl["video_path"] = df_wlasl["video_id"].astype(str).map(video_lookup)
df_wlasl["file_exists"] = df_wlasl["video_path"].notna()

print(df_wlasl["file_exists"].value_counts())
print("-------------")

df_wlasl_files = df_wlasl[df_wlasl["file_exists"]].copy()

print("Before file filtering:", len(df_wlasl))
print("After file filtering:", len(df_wlasl_files))
print("Removed:", len(df_wlasl) - len(df_wlasl_files))

print("-------------")

# =========================================================
# 8. READABILITY CHECK
# =========================================================
def can_open_video(path):
    cap = cv2.VideoCapture(path)
    ok = cap.isOpened()
    cap.release()
    return ok

df_wlasl_files["readable"] = df_wlasl_files["video_path"].apply(can_open_video)
print(df_wlasl_files["readable"].value_counts())

print("-------------")

df_wlasl_clean = df_wlasl_files[df_wlasl_files["readable"]].copy()

print("Before readability filtering:", len(df_wlasl_files))
print("After readability filtering:", len(df_wlasl_clean))

print("-------------")

# =========================================================
# 9. EXTRACT VIDEO INFO
# =========================================================
def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {
            "frames": None,
            "fps": None,
            "width": None,
            "height": None,
            "duration": None,
        }

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps if fps and fps > 0 else None
    cap.release()

    return {
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
    }

video_info = df_wlasl_clean["video_path"].apply(get_video_info)
video_info_df = pd.DataFrame(video_info.tolist())

df_wlasl_clean = pd.concat(
    [df_wlasl_clean.reset_index(drop=True), video_info_df],
    axis=1
)

print(df_wlasl_clean.head())
print("-------------")

# =========================================================
# 10. VIDEO PROPERTY STATS
# =========================================================
print(df_wlasl_clean[["frames", "fps", "width", "height", "duration"]].describe())

print("-------------")

df_wlasl_clean["duration"].hist(bins=30, figsize=(8, 5))
plt.title("Video duration distribution")
plt.xlabel("Duration (seconds)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

print("-------------")

df_wlasl_clean["frames"].hist(bins=30, figsize=(8, 5))
plt.title("Frame count distribution")
plt.xlabel("Frames")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

print("-------------")

resolution_counts = (
    df_wlasl_clean["width"].astype(str) + "x" + df_wlasl_clean["height"].astype(str)
).value_counts()

print(resolution_counts.head(10))
print("-------------")

# =========================================================
# 11. CLEAN DATASET
# =========================================================
df_wlasl_clean = df_wlasl_clean.dropna(subset=["gloss", "video_path"]).copy()
df_wlasl_clean = df_wlasl_clean[df_wlasl_clean["frames"] >= MIN_FRAMES].copy()

print(f"After MIN_FRAMES >= {MIN_FRAMES}:")
print("Clean samples:", len(df_wlasl_clean))
print("Remaining glosses:", df_wlasl_clean["gloss"].nunique())

print("-------------")

# =========================================================
# 12. CHECK DIFFERENT GLOSS THRESHOLDS
# =========================================================
gloss_counts_clean = df_wlasl_clean["gloss"].value_counts()

print("Gloss count summary after cleaning:")
print(gloss_counts_clean.describe())

print("-------------")
print("Threshold comparison:")

for threshold in [3, 5, 8, 10]:
    valid_glosses_tmp = gloss_counts_clean[gloss_counts_clean >= threshold].index
    df_tmp = df_wlasl_clean[df_wlasl_clean["gloss"].isin(valid_glosses_tmp)].copy()
    print(f"Threshold >= {threshold}: {len(df_tmp)} samples, {df_tmp['gloss'].nunique()} glosses")

print("-------------")

# =========================================================
# 13. CREATE MODEL DATASET
# =========================================================
valid_glosses = gloss_counts_clean[gloss_counts_clean >= MIN_SAMPLES_PER_GLOSS].index
df_wlasl_model = df_wlasl_clean[df_wlasl_clean["gloss"].isin(valid_glosses)].copy()

print(f"Final model subset with MIN_SAMPLES_PER_GLOSS >= {MIN_SAMPLES_PER_GLOSS}:")
print("Model samples:", len(df_wlasl_model))
print("Remaining glosses:", df_wlasl_model["gloss"].nunique())

print("-------------")

# =========================================================
# 14. SAVE OUTPUTS
# =========================================================
clean_out = PROCESSED_DIR / f"wlasl_clean_metadata_min_frames_{MIN_FRAMES}.csv"
model_out = PROCESSED_DIR / f"wlasl_model_metadata_min_frames_{MIN_FRAMES}.csv"

df_wlasl_clean.to_csv(clean_out, index=False)
df_wlasl_model.to_csv(model_out, index=False)

print(f"Saved {clean_out}")
print(f"Saved {model_out}")