import cv2
import csv
import time
import random
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_ROOT = BASE_DIR / "wlasl_processed" / "videos"
CUSTOM_ROOT = BASE_DIR  / "videos"
METADATA_CSV = BASE_DIR / "custom_metadata.csv"

SIGNER_ID = 1
FPS = 25
WIDTH = 1280
HEIGHT = 720
RECORD_SECONDS = 4
COUNTDOWN_SECONDS = 3
REFERENCE_VIDEOS_PER_ROUND = 2
SPLIT = "train"
SOURCE = "custom"

REFERENCE_WINDOW = "WLASL Reference"
RECORDER_WINDOW = "WLASL Recorder"

CUSTOM_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)

if not METADATA_CSV.exists():
    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_id", "gloss", "signer_id", "fps",
            "frame_start", "frame_end", "split", "source",
            "reference_video", "file_path"
        ])


def make_window(window_name: str):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WIDTH, HEIGHT)
    cv2.moveWindow(window_name, 80, 60)

    # Try to force window to front on supported OpenCV builds
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass


def close_window(window_name: str):
    try:
        cv2.destroyWindow(window_name)
    except Exception:
        pass


def list_reference_videos(gloss: str):
    gloss_dir = REFERENCE_ROOT / gloss
    if not gloss_dir.exists():
        return []
    return sorted(gloss_dir.glob("*.mp4"))


def list_custom_videos(gloss: str):
    gloss_dir = CUSTOM_ROOT / gloss
    if not gloss_dir.exists():
        return []
    return sorted(gloss_dir.glob("*.mp4"))


def get_next_take_index(gloss: str):
    return len(list_custom_videos(gloss)) + 1


def show_progress():
    print("\nCollected custom clips:")
    gloss_dirs = sorted([p for p in CUSTOM_ROOT.iterdir() if p.is_dir()]) if CUSTOM_ROOT.exists() else []
    if not gloss_dirs:
        print("  none yet")
        return

    found_any = False
    for gloss_dir in gloss_dirs:
        count = len(list(gloss_dir.glob("*.mp4")))
        if count > 0:
            print(f"  {gloss_dir.name}: {count}")
            found_any = True

    if not found_any:
        print("  none yet")


def draw_text_block(frame, lines, start_y=60, line_gap=55, scale=1.2, color=(255, 255, 255), thickness=2):
    y = start_y
    for line in lines:
        cv2.putText(
            frame,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        y += line_gap


def play_reference_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open reference video: {video_path}")
        return "next"

    make_window(REFERENCE_WINDOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (1240, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        draw_text_block(
            frame,
            [
                f"Reference: {video_path.name}",
                "R = record   N = next   SPACE = replay   Q = quit"
            ],
            start_y=55,
            line_gap=45,
            scale=1.0
        )

        cv2.imshow(REFERENCE_WINDOW, frame)
        key = cv2.waitKey(int(1000 / FPS)) & 0xFF

        if key == ord("r"):
            cap.release()
            close_window(REFERENCE_WINDOW)
            return "record"
        elif key == ord("n"):
            cap.release()
            close_window(REFERENCE_WINDOW)
            return "next"
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return "quit"
        elif key == 32:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


def countdown(cam, gloss: str):
    make_window(RECORDER_WINDOW)
    start = time.time()

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        elapsed = time.time() - start
        remaining = max(0.0, COUNTDOWN_SECONDS - elapsed)

        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (760, 220), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        draw_text_block(
            frame,
            [
                f"Gloss: {gloss}",
                f"Starting in: {remaining:0.1f}s",
                "Press Q to quit"
            ],
            start_y=60,
            line_gap=55,
            scale=1.2,
            color=(255, 255, 255),
            thickness=3
        )

        cv2.imshow(RECORDER_WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return False

        if elapsed >= COUNTDOWN_SECONDS:
            return True


def record_clip(cam, gloss: str, take_idx: int, reference_name: str):
    gloss_dir = CUSTOM_ROOT / gloss
    gloss_dir.mkdir(parents=True, exist_ok=True)

    video_id = f"{gloss}_s{SIGNER_ID:02d}_{take_idx:03d}"
    out_path = gloss_dir / f"{video_id}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (WIDTH, HEIGHT))

    make_window(RECORDER_WINDOW)
    start = time.time()
    frame_count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        elapsed = time.time() - start
        remaining = max(0.0, RECORD_SECONDS - elapsed)

        display = frame.copy()
        overlay = display.copy()
        cv2.rectangle(overlay, (15, 15), (760, 240), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.45, display, 0.55, 0)

        draw_text_block(
            display,
            [
                f"REC: {gloss}",
                f"Take: {take_idx}",
                f"Time left: {remaining:0.1f}s"
            ],
            start_y=60,
            line_gap=55,
            scale=1.2,
            color=(255, 255, 255),
            thickness=3
        )

        cv2.imshow(RECORDER_WINDOW, display)
        writer.write(frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            writer.release()
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            return None

        if elapsed >= RECORD_SECONDS:
            break

    writer.release()

    with open(METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            video_id,
            gloss,
            SIGNER_ID,
            FPS,
            1,
            frame_count,
            SPLIT,
            SOURCE,
            reference_name,
            str(out_path)
        ])

    return out_path


def choose_gloss():
    while True:
        user_input = input("\nEnter gloss name ('progress', 'q'): ").strip().lower()

        if user_input == "q":
            return None
        elif user_input == "progress":
            show_progress()
        elif user_input == "":
            print("Please enter a gloss.")
        else:
            return user_input


def main():
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"REFERENCE_ROOT: {REFERENCE_ROOT}")
    print(f"CUSTOM_ROOT: {CUSTOM_ROOT}")
    print(f"METADATA_CSV: {METADATA_CSV}")

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cam.set(cv2.CAP_PROP_FPS, FPS)

    if not cam.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        while True:
            gloss = choose_gloss()
            if gloss is None:
                break

            ref_videos = list_reference_videos(gloss)
            custom_count = len(list_custom_videos(gloss))

            print(f"\nGloss: {gloss}")
            print(f"Reference videos found: {len(ref_videos)}")
            print(f"Your custom clips so far: {custom_count}")

            selected_reference = ""

            if ref_videos:
                chosen_refs = random.sample(
                    ref_videos,
                    min(REFERENCE_VIDEOS_PER_ROUND, len(ref_videos))
                )

                should_record = False

                for ref in chosen_refs:
                    action = play_reference_video(ref)
                    selected_reference = ref.name

                    if action == "record":
                        should_record = True
                        break
                    elif action == "quit":
                        return

                if not should_record:
                    go_on = input("Record anyway? (y/n): ").strip().lower()
                    if go_on != "y":
                        continue
            else:
                print("No reference videos available for this gloss.")
                go_on = input("Record this gloss anyway? (y/n): ").strip().lower()
                if go_on != "y":
                    continue

            while True:
                take_idx = get_next_take_index(gloss)

                ok = countdown(cam, gloss)
                if not ok:
                    return

                saved = record_clip(cam, gloss, take_idx, selected_reference)
                if saved is None:
                    return

                print(f"Saved: {saved}")

                again = input("Another take for this gloss? (y/n): ").strip().lower()
                if again != "y":
                    break

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()