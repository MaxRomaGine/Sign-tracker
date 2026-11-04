# Sign Tracker

Sign Tracker is a local webcam prototype for recognizing a small, user-recorded vocabulary of hand signs. It uses MediaPipe to find hands and a Random Forest to classify a one-second movement window. It is a learning project, not a general sign-language translator.

## Platform support

The pinned MediaPipe 0.10.21 build has Python 3.10–3.12 wheels for Windows x64 and macOS Intel and Apple Silicon. The pinned OpenCV build needs macOS 13 or newer. Windows ARM is not covered by this pinned build. A built-in laptop webcam is easiest to start with; external cameras can be selected with `--camera 1`, `--camera 2`, and so on.

## Install

Use Python 3.11 if possible. Create the environment in the project folder, install the pinned dependencies, and download the official MediaPipe hand model once.

### macOS (Intel or Apple Silicon)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.setup
```

On first use, allow camera access when macOS asks. If you denied it earlier, enable it under **System Settings → Privacy & Security → Camera**.

### Windows (PowerShell, x64)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.setup
```

The `.venv\Scripts\python.exe` commands work without changing PowerShell's script-execution policy. In Windows Settings, allow desktop apps to use the camera. Close video-call apps if they already have it open.

## Check the camera

Run this before recording. It opens the camera, captures one frame, and exits:

```bash
python -m src.debug --check
```

On Windows, replace `python` with `.\.venv\Scripts\python.exe`. On macOS, use the `python` command from the activated `.venv`. If camera 0 fails, try `--camera 1`. To see the camera and live landmark activity, run `python -m src.debug` and press `q` to close it.

## Record examples

Each Space press records a one-second window. For a still sign, hold the hand shape steady in view. For a moving sign, perform the whole movement during the one-second capture. Vary your distance, position, and speed across recordings.

Record each label in several separate runs. Each run is saved as a separate session, which lets training hold out whole sessions for a more honest score. Record each class at least three times, closing and rerunning the command between sessions.

```bash
python -m src.collect --label hola --samples 20
python -m src.collect --label gracias --samples 20
python -m src.collect --label idle --samples 20
```

Repeat each command in new runs to create separate sessions. For `idle`, record a mix of hands out of view and ordinary non-sign hand movement; the prompt reminds you what to capture. Idle examples help the model learn when not to show a sign. Record idle data in multiple sessions too. A useful starting dataset is 3–5 sessions per sign, with 20–30 examples per session.

The dataset is stored in `data/signs.npz`; the trained model is `models/sign_classifier.pkl`. Both stay on your computer and are ignored by Git. Webcam-derived landmarks can still be personal data, so share them only intentionally.

## Train and recognize

```bash
python -m src.train
python -m src.infer
```

The recognizer runs MediaPipe on CPU for predictable cross-platform behavior. It combines motion with a wrist-centered, scale-normalized hand-shape description. It samples camera frames onto a fixed one-second timeline, so slower and faster camera frame rates use the same feature window. It requires several consistent predictions before showing a sign and holds the result briefly. The `idle` class can suppress false activations when its examples cover ordinary movement well.

Training reports a session holdout score when every label has samples from at least two separate collection sessions. It holds out one complete session per label, then trains the final model on all examples. The score is useful for spotting obvious problems, but it still reflects one signer, camera, and environment. Test with new sessions and ordinary movements before trusting it.

Useful options:

```bash
python -m src.collect --label hola --samples 25 --camera 1
python -m src.train --data data/signs.npz --output models/my_model.pkl --trees 300
python -m src.infer --model models/my_model.pkl --camera 1 --threshold 0.65
```

On Windows, use `.\.venv\Scripts\python.exe -m ...` in place of `python -m ...` if you did not activate the environment.

## Troubleshooting

- **MediaPipe model missing:** run `python -m src.setup` from the project folder.
- **Camera will not open:** run `python -m src.debug --check`, allow camera access, close other camera apps, and try `--camera 1`.
- **No signs appear:** retrain after collecting multiple sessions per class; ensure hands are well lit and visible; lower `--threshold` slightly only if false activations remain low.
- **Old dataset or model rejected:** the feature format changed. Move old files aside and record fresh examples with this version.
- **Install fails:** use 64-bit Python 3.11 and recreate the virtual environment. MediaPipe does not provide a wheel for every Python version or computer architecture.

## What it can and cannot recognize

- It can learn a small personal set of static hand shapes and short movements from recorded examples.
- Hand landmarks alone do not capture facial expression, body grammar, or continuous signing.
- Similar signs, occlusion, changes in signer or lighting, and spontaneous movement can cause errors.
- Results are not validated for communication, accessibility, or use in safety-critical situations.

## Project layout

```text
src/
  collect.py    Record labeled one-second camera windows and session IDs
  extractor.py  Detect hands and keep left/right landmark slots stable
  utils.py      Resample time windows and build motion plus hand-shape features
  train.py      Validate data, evaluate by session, and train a classifier
  infer.py      Run confidence-gated live recognition
  debug.py      Check camera access or preview landmark activity
  setup.py      Download the required MediaPipe task model
```

## Next improvements

The next useful step is a small evaluation set recorded on different days and, with consent, by another signer. Track both missed signs and false activations. If the project grows beyond a personal vocabulary, explore a temporal model and a sign-language dataset developed with Deaf language experts.
