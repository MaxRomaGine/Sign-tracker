"""Run live webcam inference using a model produced by src.train."""

import argparse
from collections import deque
from pathlib import Path
import time

import cv2
import joblib
import numpy as np

from src.collect import open_camera
from src.extractor import HandExtractor
from src.utils import FEATURE_COUNT, FEATURE_SCHEMA_VERSION, WINDOW_SECONDS, extract_kinematic_features, resample_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/sign_classifier.pkl"), help="Trained model")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--threshold", type=float, default=0.60, help="Minimum prediction confidence")
    parser.add_argument("--hold", type=float, default=1.2, help="Seconds to keep the last recognized sign visible")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.hold < 0:
        parser.error("--hold cannot be negative")
    if not args.model.exists():
        parser.error(f"Model not found: {args.model}. Record examples and run python -m src.train first.")

    model = joblib.load(args.model)
    if getattr(model, "_sign_tracker_feature_version", None) != FEATURE_SCHEMA_VERSION:
        parser.error("Model uses an older feature format. Record fresh data and run python -m src.train.")
    if getattr(model, "n_features_in_", None) != FEATURE_COUNT:
        parser.error("Model feature count is incompatible; retrain it with the current project version.")
    landmarker_path = Path("models/hand_landmarker.task")
    if not landmarker_path.exists():
        parser.error(f"MediaPipe model not found: {landmarker_path}. Follow the setup instructions in README.md.")
    extractor = HandExtractor()
    try:
        cap = open_camera(args.camera)
    except RuntimeError as exc:
        extractor.close()
        parser.error(str(exc))
    frame_history = deque()
    recent_predictions = deque(maxlen=5)
    last_label = ""
    display_until = 0.0
    print(f"Camera: {cap.getBackendName()} | resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    print("Live recognition started. Press q to quit. Hold static signs clearly or perform dynamic signs naturally.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                break
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            vector = extractor.extract(frame)
            frame_history.append((now, vector))
            cutoff = now - WINDOW_SECONDS
            while len(frame_history) > 2 and frame_history[1][0] <= cutoff:
                frame_history.popleft()
            confidence = 0.0
            predicted = ""

            if len(frame_history) >= 2 and frame_history[0][0] <= cutoff:
                timestamps, vectors = zip(*frame_history)
                sampled = resample_window(vectors, timestamps)
                hands_present = np.max(np.abs(sampled)) > 1e-4
                has_idle_class = "idle" in getattr(model, "classes_", [])
                if hands_present or has_idle_class:
                    features = extract_kinematic_features(sampled).reshape(1, -1)
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(features)[0]
                        index = int(np.argmax(probabilities))
                        predicted = str(model.classes_[index])
                        confidence = float(probabilities[index])
                    else:
                        predicted = str(model.predict(features)[0])
                        confidence = 1.0
                    if confidence >= args.threshold and predicted != "idle":
                        recent_predictions.append(predicted)
                        if len(recent_predictions) >= 3 and max(set(recent_predictions), key=recent_predictions.count) == predicted:
                            last_label = predicted
                            display_until = now + args.hold
                    else:
                        recent_predictions.clear()
                else:
                    recent_predictions.clear()
            else:
                recent_predictions.clear()

            label = last_label if now <= display_until else ""
            cv2.putText(frame, f"Sign: {label or '—'}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (40, 220, 40) if label else (190, 190, 190), 2)
            if predicted:
                cv2.putText(frame, f"Candidate: {predicted} ({confidence:.0%})", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            cv2.putText(frame, "q: quit", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (220, 220, 220), 1)
            cv2.imshow("Sign Tracker - Live", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
