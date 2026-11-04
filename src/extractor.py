"""Extract a stable, camera-relative representation of up to two hands."""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandExtractor:
    """Return 126 float features: 21 xyz landmarks for each hand.

    Hand slots are left/right from the camera user's perspective, using
    MediaPipe's handedness metadata. A missing hand is represented by zeros.
    Coordinates use MediaPipe's normalized camera coordinates. MediaPipe Tasks
    handedness metadata keeps the two hand slots consistent across frames.
    """

    def __init__(self, model_path: str = "models/hand_landmarker.task", max_num_hands: int = 2):
        self.max_num_hands = max_num_hands
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU,
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def extract(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.landmarker.detect(image)
        vector = np.zeros((2, 21, 3), dtype=np.float32)

        if results.hand_landmarks:
            for index, hand in enumerate(results.hand_landmarks[: self.max_num_hands]):
                side = "Left"
                if index < len(results.handedness) and results.handedness[index]:
                    side = results.handedness[index][0].category_name
                slot = 0 if side == "Left" else 1
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
                vector[slot] = coords

        return vector.reshape(-1)

    def close(self) -> None:
        self.landmarker.close()
