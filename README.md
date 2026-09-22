# Sign Tracker

Real-time sign language recognition system built with Python, OpenCV, MediaPipe, **Piecewise Relative Kinematics**, and a Random Forest classifier. Designed for zero-flicker live inference, strict idle rejection, and instant sign override.

---

## Quickstart (Plug & Play)

If you want to test it right away with a pre-trained model:

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/MaxRomaGine/Sign-tracker.git](https://github.com/MaxRomaGine/Sign-tracker.git)
   cd Sign-tracker
1. Create virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Download the pre-trained model (from GitHub Releases):
   ```Bash
   mkdir -p models
   curl -L "[https://github.com/MaxRomaGine/Sign-tracker/releases/download/v1.0/sign_classifier.pkl](https://github.com/MaxRomaGine/Sign-tracker/releases/download/v1.0/sign_classifier.pkl)" -o models/sign_classifier.pkl

3. Run live inference:
   ```Bash
   python3 -m src.infer
   (Press q to exit)

4. Train Your Own Signs (Custom Dataset)
If you prefer recording your own vocabulary from scratch:

4.1 Record samples:
   ```Bash  
   python3 -m src.collect --label hola --samples 35
   python3 -m src.collect --label gracias --samples 35
   python3 -m src.collect --label idle --samples 40
   ```
   Idle means when no sing is performed.
   
4.2 Train the model:
   ```Bash
   python3 -m src.train
   ```
4.3 Run inference:
   ```Bash
   python3 -m src.infer
   ```
## Architecture & Design Decisions:

- Spatial Invariance: Does not rely on absolute screen coordinates; processes piecewise relative kinematics (378 dimensions) to avoid position cheating.

- Idle Gating: Explicit idle/free-movement class to keep the screen completely clean when no real sign is present.

- Temporal Latch & Override: 2-second screen hold with instant override if a valid trajectory change is detected.

## Project Structure:
```
Sign-tracker/
├── src/
│   ├── extractor.py    # MediaPipe landmark extraction
│   ├── utils.py        # Piecewise relative kinematic feature engineering
│   ├── collect.py      # Dataset collection CLI
│   ├── train.py        # Random Forest training pipeline
│   └── infer.py        # Real-time inference loop
├── requirements.txt
└── README.md
```
