import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from src.utils import extract_kinematic_features

def main():
    data_path = "data/signs.npz"
    if not os.path.exists(data_path):
        print("[-] No se encontró el dataset en data/signs.npz")
        return
        
    data = np.load(data_path, allow_pickle=True)
    X_raw, y = data["X"], data["y"]
    
    N = X_raw.shape[0]
    X_rel = np.array([extract_kinematic_features(sample.reshape(30, 126)) for sample in X_raw], dtype=np.float32)
    
    print(f"[+] Loaded dataset. Rel X shape: {X_rel.shape}, classes: {np.unique(y)}")
    
    test_sz = 0.2 if N >= 5 else 0.5
    X_train, X_test, y_train, y_test = train_test_split(X_rel, y, test_size=test_sz, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.ravel() if y_train.ndim > 1 else y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[+] Accuracy con cinemática relativa: {acc:.2f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sign_classifier.pkl")
    print("[+] Modelo guardado en models/sign_classifier.pkl")

if __name__ == "__main__":
    main()