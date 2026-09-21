import os
import cv2
import numpy as np
import joblib
import time
from collections import deque
from src.extractor import HandExtractor
from src.utils import extract_kinematic_features

def main():
    model_path = "models/sign_classifier.pkl"
    if not os.path.exists(model_path):
        print("[-] No hay modelo entrenado. Ejecuta train.py primero.")
        return
        
    model = joblib.load(model_path)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    extractor = HandExtractor()
    
    window_size = 30
    frame_buffer = deque(maxlen=window_size)
    confidence_threshold = 0.52  

    active_label = ""
    display_until = 0.0
    hold_duration = 2.0  # 2 segundos clavados
    
    print("[+] Real-time inference ON. Press 'q' to exit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        display = frame.copy()
        vec = extractor.extract(frame)
        frame_buffer.append(vec)
        
        hands_active = np.sum(np.abs(vec)) > 1e-4
        now = time.time()
        
       # Evaluar solo si hay actividad y ventana llena
        if hands_active and len(frame_buffer) == window_size:
            buffer_mat = np.array(list(frame_buffer))
            feat_vec = extract_kinematic_features(buffer_mat).reshape(1, -1)
            
            if np.max(np.abs(feat_vec)) > 0.005:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(feat_vec)[0]
                    max_prob = np.max(probs)
                    pred_idx = np.argmax(probs)
                    pred_class = model.classes_[pred_idx]
                    
                    # Interrupción inmediata si supera el umbral y no es idle
                    if max_prob >= confidence_threshold and pred_class != "idle":
                        active_label = pred_class
                        display_until = now + hold_duration
                else:
                    pred_class = model.predict(feat_vec)[0]
                    if pred_class != "idle":
                        active_label = pred_class
                        display_until = now + hold_duration

        # Mostrar solo si estamos dentro de la ventana de tiempo activa
        current_text = active_label if now <= display_until else ""
            
        if current_text:
            cv2.putText(display, f"Sign: {current_text}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
        cv2.imshow("Sign Tracker - Live", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()