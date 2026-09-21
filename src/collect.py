import os
import cv2
import numpy as np
import argparse
from src.extractor import HandExtractor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True, help="Nombre de la seña (ej: hola)")
    parser.add_argument("--samples", type=int, default=15, help="Número de muestras a grabar")
    args = parser.parse_args()

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    extractor = HandExtractor()
    
    os.makedirs("data", exist_ok=True)
    data_path = "data/signs.npz"
    
    X_all, y_all = [], []
    if os.path.exists(data_path):
        existing = np.load(data_path, allow_pickle=True)
        X_all, y_all = list(existing["X"]), list(existing["y"])

    print(f"[+] Recording for class '{args.label}'. Target samples: {args.samples}")
    print("[!] Press 'r' to start recording a sample, 'q' to quit.")

    sample_count = 0
    while sample_count < args.samples and cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        display = frame.copy()
        cv2.putText(display, f"Class: {args.label} | Sample: {sample_count}/{args.samples}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Press 'r' to record 1s window", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Collector", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print(f"[+] Recording sample {sample_count+1}...")
            buffer = []
            for _ in range(30):
                ret_f, f_frame = cap.read()
                if not ret_f: break
                feats = extractor.extract(f_frame)
                buffer.append(feats)
                
                fb = f_frame.copy()
                cv2.putText(fb, "RECORDING...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow("Collector", fb)
                cv2.waitKey(30)
            
            if len(buffer) == 30:
                sample_vector = np.concatenate(buffer).flatten()  # 30 * 126 = 3780
                X_all.append(sample_vector)
                y_all.append(args.label)
                sample_count += 1
                print(f"[+] Saved sample {sample_count}")

    cap.release()
    cv2.destroyAllWindows()

    if X_all:
        np.savez(data_path, X=np.array(X_all, dtype=np.float32), y=np.array(y_all))
        print(f"[+] Dataset updated at {data_path}. Total samples: {len(y_all)}")

if __name__ == "__main__":
    main()