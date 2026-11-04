"""Preview camera landmark activity while troubleshooting setup."""

import argparse

import cv2
import numpy as np

from src.collect import open_camera
from src.extractor import HandExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()
    try:
        cap = open_camera(args.camera)
    except RuntimeError as exc:
        parser.error(str(exc))
    extractor = HandExtractor()
    print("Camera feature preview started. Press q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                break
            frame = cv2.flip(frame, 1)
            vector = extractor.extract(frame)
            count = int(np.count_nonzero(vector))
            text = f"Active landmark values: {count}/126"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 40), 2)
            cv2.imshow("Sign Tracker - Camera Debug", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
