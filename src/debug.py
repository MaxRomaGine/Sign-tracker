import cv2
import numpy as np
from src.extractor import HandExtractor

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    extractor = HandExtractor()

    print("[+] Debug mode ON. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        display = frame.copy()
        vector = extractor.extract(frame)
        
        # Si el vector tiene chicha (no todo ceros), hay detección activa
        active_hands = int(np.sum(vector != 0) > 0)
        status_text = f"Vector active features: {np.count_nonzero(vector)}/126"

        cv2.putText(display, status_text, (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Debug - Pipeline Validation", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()