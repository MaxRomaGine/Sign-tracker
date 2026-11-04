"""Download the MediaPipe hand model into the project-local models folder."""

import argparse
from pathlib import Path
import urllib.error
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path("models/hand_landmarker.task")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Download again even if the model already exists")
    args = parser.parse_args()
    if MODEL_PATH.exists() and not args.force:
        print(f"MediaPipe model already exists: {MODEL_PATH}")
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".download")
    try:
        print("Downloading the MediaPipe hand model...")
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, temporary_path.open("wb") as output:
            output.write(response.read())
        if temporary_path.stat().st_size < 1_000_000:
            raise ValueError("Downloaded file is unexpectedly small; check the network connection and retry.")
        temporary_path.replace(MODEL_PATH)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        temporary_path.unlink(missing_ok=True)
        parser.error(f"Could not download the MediaPipe model: {exc}")
    print(f"MediaPipe model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
