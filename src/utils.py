"""Shared feature engineering for collection, training, and inference."""

import numpy as np

WINDOW_SIZE = 30
FRAME_FEATURES = 126
WINDOW_SECONDS = 1.0
KINEMATIC_FEATURES = FRAME_FEATURES * 3
POSE_FEATURES = FRAME_FEATURES
FEATURE_COUNT = KINEMATIC_FEATURES + POSE_FEATURES
FEATURE_SCHEMA_VERSION = 3


def resample_window(frames, timestamps, duration=WINDOW_SECONDS, output_frames=WINDOW_SIZE):
    """Interpolate irregular camera frames onto a fixed-duration time grid."""
    values = np.asarray(frames, dtype=np.float32)
    times = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FRAME_FEATURES or len(values) != len(times):
        raise ValueError("Frames must have shape (n, 126) and one timestamp per frame")
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("At least two frames with increasing timestamps are required")
    start, end = times[0], times[-1]
    if end - start < duration:
        raise ValueError(f"Frames span {end - start:.2f}s; at least {duration:.2f}s is required")
    target_times = np.linspace(end - duration, end, output_frames)
    sampled = np.stack([np.interp(target_times, times, values[:, column]) for column in range(FRAME_FEATURES)], axis=1)
    return sampled.astype(np.float32)


def extract_kinematic_features(window_matrix: np.ndarray) -> np.ndarray:
    """Combine motion features with a scale-normalized static hand shape."""
    window = np.asarray(window_matrix, dtype=np.float32)
    if window.shape != (WINDOW_SIZE, FRAME_FEATURES):
        raise ValueError(
            f"Expected a ({WINDOW_SIZE}, {FRAME_FEATURES}) window, got {window.shape}"
        )
    middle = WINDOW_SIZE // 2
    first_half = window[middle] - window[0]
    second_half = window[-1] - window[middle]
    peak_velocity = np.max(np.abs(np.diff(window, axis=0)), axis=0)

    # Add a stable hand-shape descriptor so stationary signs are recognizable.
    hand_frames = window.reshape(WINDOW_SIZE, 2, 21, 3)
    hand_shapes = np.zeros((2, 21, 3), dtype=np.float32)
    for hand_index in range(2):
        hand = hand_frames[:, hand_index]
        present = np.any(np.abs(hand) > 1e-6, axis=(1, 2))
        if not np.any(present):
            continue
        landmarks = np.median(hand[present], axis=0)
        centered = landmarks - landmarks[0]  # wrist-relative hand shape
        palm_width = float(np.linalg.norm(landmarks[5] - landmarks[17]))
        if palm_width <= 1e-5:
            palm_width = float(np.max(np.linalg.norm(centered, axis=1)))
        if palm_width > 1e-5:
            hand_shapes[hand_index] = centered / palm_width

    return np.concatenate((first_half, second_half, peak_velocity, hand_shapes.reshape(-1))).astype(np.float32)
