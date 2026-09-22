import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandExtractor:
    def __init__(self, model_path: str = "models/hand_landmarker.task", max_num_hands: int = 2):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.max_hands = max_num_hands

    def extract(self, frame: np.ndarray) -> np.ndarray:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = self.landmarker.detect(mp_image)
        output_vector = np.zeros(self.max_hands * 21 * 3, dtype=np.float32)
        
        if results.hand_landmarks:
            for idx, hand in enumerate(results.hand_landmarks[:self.max_hands]):
                hand_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32).flatten()
                start_idx = idx * 21 * 3
                output_vector[start_idx:start_idx + len(hand_coords)] = hand_coords
                
        return output_vector

    def close(self):
        self.landmarker.close()