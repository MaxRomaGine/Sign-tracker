"""Shared feature engineering for collection, training, and inference."""

import numpy as np

WINDOW_SIZE = 30
FRAME_FEATURES = 126
KINEMATIC_FEATURES = FRAME_FEATURES * 3
FEATURE_SCHEMA_VERSION = 2


def extract_kinematic_features(window_matrix: np.ndarray) -> np.ndarray:
    """Summarize a frame window using two displacements and peak velocity."""
    window = np.asarray(window_matrix, dtype=np.float32)
    if window.shape != (WINDOW_SIZE, FRAME_FEATURES):
        raise ValueError(
            f"Expected a ({WINDOW_SIZE}, {FRAME_FEATURES}) window, got {window.shape}"
        )
    middle = WINDOW_SIZE // 2
    first_half = window[middle] - window[0]
    second_half = window[-1] - window[middle]
    peak_velocity = np.max(np.abs(np.diff(window, axis=0)), axis=0)
    return np.concatenate((first_half, second_half, peak_velocity)).astype(np.float32)
