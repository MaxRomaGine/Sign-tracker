"""Record labeled one-second webcam windows for the local dataset."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.extractor import HandExtractor
from src.utils import FEATURE_SCHEMA_VERSION, FRAME_FEATURES, WINDOW_SIZE


def open_camera(index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera {index}. Check permissions or try --camera N.")
    return cap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Sign label, for example hello")
    parser.add_argument("--samples", type=int, default=20, help="Number of windows to record")
    parser.add_argument("--data", type=Path, default=Path("data/signs.npz"), help="Dataset path")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()
    if not args.label.strip() or args.label.strip().lower() == "idle":
        parser.error("Use a non-empty sign label. Idle is handled as no detected hand activity.")
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
            old_x, old_y = saved["X"], saved["y"].astype(str)
        if old_x.ndim != 2 or old_x.shape[1] != WINDOW_SIZE * FRAME_FEATURES or len(old_x) != len(old_y):
            parser.error(f"Existing dataset at {args.data} has an incompatible shape")
        X_all, y_all = list(old_x), list(old_y)
    else:
        X_all, y_all = [], []

    cap = open_camera(args.camera)
    extractor = HandExtractor()
    sample_count = 0
    frames = []
    recording = False
    try:
        print("Show the sign clearly, then press SPACE to record a one-second window. Press q to finish.")
        while sample_count < args.samples:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                break
            frame = cv2.flip(frame, 1)
            if recording:
                frames.append(extractor.extract(frame))
                if len(frames) >= WINDOW_SIZE:
                    X_all.append(np.asarray(frames[:WINDOW_SIZE], dtype=np.float32).reshape(-1))
                    y_all.append(label)
                    sample_count += 1
                    frames.clear()
                    recording = False
                    print(f"Saved {sample_count}/{args.samples}")

            display = frame.copy()
            instruction = "RECORDING - hold the sign" if recording else "SPACE: record sample | q: quit"
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
                print("Recording sample...")
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()

    if X_all:
        np.savez_compressed(args.data, X=np.asarray(X_all, dtype=np.float32), y=np.asarray(y_all, dtype=str),
                            feature_version=np.array(FEATURE_SCHEMA_VERSION, dtype=np.int32))
        print(f"Dataset saved to {args.data} ({len(y_all)} total samples).")
    print(f"Recorded {sample_count} new samples for '{label}'.")


if __name__ == "__main__":
    main()
