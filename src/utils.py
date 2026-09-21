import numpy as np

def extract_kinematic_features(window_matrix):
    # window_matrix shape: (30, 126)
    mid_idx = window_matrix.shape[0] // 2
    start_frame = window_matrix[0]
    mid_frame = window_matrix[mid_idx]
    end_frame = window_matrix[-1]
    
    disp_1 = mid_frame - start_frame       # Tramo 1: primera mitad (126 dims)
    disp_2 = end_frame - mid_frame         # Tramo 2: segunda mitad (126 dims)
    frame_diffs = np.diff(window_matrix, axis=0)
    max_velocity = np.max(np.abs(frame_diffs), axis=0) # Picos de velocidad (126 dims)
    
    return np.concatenate([disp_1, disp_2, max_velocity]) # 378 dims total