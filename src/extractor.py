import cv2
import mediapipe as mp
import numpy as np

class HandExtractor:
    def __init__(self, max_num_hands: int = 2):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb)
        torso_origin = np.zeros(3, dtype=np.float32)
        scale_ref = 1.0
        scale_found = False

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            l_sh = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z], dtype=np.float32)
            r_sh = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z], dtype=np.float32)
            torso_origin = (l_sh + r_sh) / 2.0
            dist = np.linalg.norm(l_sh - r_sh)
            if dist > 1e-4:
                scale_ref = dist
                scale_found = True

        hands_results = self.hands.process(rgb)
        vector = np.zeros(126, dtype=np.float32)

        if hands_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks[:2]):
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                if scale_found:
                    norm_coords = (coords - torso_origin) / scale_ref
                else:
                    wrist = coords[0]
                    norm_coords = coords - wrist
                    max_d = np.max(np.linalg.norm(norm_coords, axis=1))
                    if max_d > 1e-6:
                        norm_coords /= max_d
                start = idx * 63
                vector[start:start+63] = norm_coords.flatten()

        return vector