"""Record labeled one-second webcam windows for the local dataset."""

import argparse
from pathlib import Path
import sys
import time
import uuid

import cv2
import numpy as np

from src.extractor import HandExtractor
from src.utils import FEATURE_SCHEMA_VERSION, FRAME_FEATURES, WINDOW_SECONDS, WINDOW_SIZE, resample_window


def open_camera(index: int):
    preferred_backend = {
        "darwin": cv2.CAP_AVFOUNDATION,
        "win32": cv2.CAP_DSHOW,
    }.get(sys.platform)
    attempts = [preferred_backend, cv2.CAP_ANY] if preferred_backend is not None else [cv2.CAP_ANY]
    for backend in attempts:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    platform_hint = "On macOS, allow camera access in System Settings > Privacy & Security > Camera." if sys.platform == "darwin" else "Close other apps using the camera and check Windows camera privacy settings."
    raise RuntimeError(f"Could not open camera {index}. {platform_hint} Try another index with --camera N.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Sign label, for example hello")
    parser.add_argument("--samples", type=int, default=20, help="Number of windows to record")
    parser.add_argument("--data", type=Path, default=Path("data/signs.npz"), help="Dataset path")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()
    if not args.label.strip():
        parser.error("Use a non-empty sign label.")
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    label = args.label.strip().lower()

    args.data.parent.mkdir(parents=True, exist_ok=True)
    if args.data.exists():
        with np.load(args.data, allow_pickle=False) as saved:
            if not {"X", "y"}.issubset(saved.files):
                parser.error(f"{args.data} must contain X and y arrays")
            if "feature_version" not in saved.files or int(saved["feature_version"]) != FEATURE_SCHEMA_VERSION:
                parser.error(f"{args.data} uses an older feature format. Back it up and record fresh samples.")
            if "session_id" not in saved.files:
                parser.error(f"{args.data} is missing session IDs; move it aside and record fresh samples")
            old_x, old_y = saved["X"], saved["y"].astype(str)
            old_sessions = saved["session_id"].astype(str)
        if old_x.ndim != 2 or old_x.shape[1] != WINDOW_SIZE * FRAME_FEATURES or len(old_x) != len(old_y):
            parser.error(f"Existing dataset at {args.data} has an incompatible shape")
        if len(old_sessions) != len(old_y):
            parser.error(f"Existing dataset at {args.data} has mismatched session IDs")
        X_all, y_all, session_ids = list(old_x), list(old_y), list(old_sessions)
    else:
        X_all, y_all, session_ids = [], [], []

    model_path = Path("models/hand_landmarker.task")
    if not model_path.exists():
        parser.error(f"MediaPipe model not found: {model_path}. Follow the setup instructions in README.md.")
    extractor = HandExtractor()
    try:
        cap = open_camera(args.camera)
    except RuntimeError as exc:
        extractor.close()
        parser.error(str(exc))
    sample_count = 0
    frames = []
    timestamps = []
    recording = False
    session_id = uuid.uuid4().hex
    try:
        print(f"Camera: {cap.getBackendName()} | resolution: {int(cap.get(3))}x{int(cap.get(4))}")
        prompt = (
            "Show no sign: mix hands-out-of-view and casual, non-sign movement"
            if label == "idle" else f"Show '{label}' clearly"
        )
        print(f"{prompt}, then press SPACE to record a {WINDOW_SECONDS:g}-second window. Press q to finish.")
        while sample_count < args.samples:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                break
            now = time.monotonic()
            frame = cv2.flip(frame, 1)
            if recording:
                frames.append(extractor.extract(frame))
                timestamps.append(now)
                if now - timestamps[0] >= WINDOW_SECONDS:
                    if len(frames) < 8:
                        print("Skipped sample: camera delivered too few frames. Check camera performance and try again.")
                    else:
                        window = resample_window(frames, timestamps)
                        X_all.append(window.reshape(-1))
                        y_all.append(label)
                        session_ids.append(session_id)
                        sample_count += 1
                        print(f"Saved {sample_count}/{args.samples}")
                    frames.clear()
                    timestamps.clear()
                    recording = False

            display = frame.copy()
            if recording and label == "idle":
                instruction = "RECORDING - rest naturally or keep hands out of view"
            else:
                instruction = "RECORDING - hold/perform the sign" if recording else "SPACE: record sample | q: quit"
            color = (0, 0, 255) if recording else (40, 220, 40)
            cv2.putText(display, f"Label: {label}  Samples: {sample_count}/{args.samples}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(display, instruction, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("Sign Tracker - Data Collection", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and not recording:
                recording = True
                frames = []
                timestamps = []
                print("Recording sample...")
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()

    if X_all:
        np.savez_compressed(args.data, X=np.asarray(X_all, dtype=np.float32), y=np.asarray(y_all, dtype=str),
                            session_id=np.asarray(session_ids, dtype=str),
                            feature_version=np.array(FEATURE_SCHEMA_VERSION, dtype=np.int32))
        print(f"Dataset saved to {args.data} ({len(y_all)} total samples).")
    print(f"Recorded {sample_count} new samples for '{label}'.")


if __name__ == "__main__":
    main()
