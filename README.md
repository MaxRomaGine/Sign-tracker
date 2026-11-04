# Sign Tracker

Sign Tracker is a local webcam prototype for recognizing a small, user-recorded vocabulary of hand signs. It uses MediaPipe to find hands and a Random Forest to classify a one-second movement window. It is a learning project, not a general sign-language translator.

## Platform support

The pinned MediaPipe 0.10.21 build has Python 3.10–3.12 wheels for Windows x64 and macOS Intel and Apple Silicon. The pinned OpenCV build needs macOS 13 or newer. Windows ARM is not covered by this pinned build. A built-in laptop webcam is easiest to start with; external cameras can be selected with `--camera 1`, `--camera 2`, and so on.

## Try the browser demo

The repository also includes a browser-based companion demo in [`web/`](web/). Type any label you want, record an example, and repeat with as many labels as your browser's local storage allows. It uses MediaPipe hand landmarks in the browser and compares live one-second hand movements against those examples. You can delete examples for one label or clear them all. The browser demo is a separate lightweight prototype; it does not load the Python Random Forest model.

To publish it on GitHub Pages:

1. In the GitHub repository, open **Settings → Pages**.
2. Under **Build and deployment**, select **GitHub Actions** as the source.
3. Push a change to the `web/` directory on `main`, or run **Deploy browser demo to GitHub Pages** from the **Actions** tab.
4. Once the workflow finishes, open the Pages URL shown in the workflow summary (usually `https://maxromagine.github.io/Sign-tracker/`).

Use a current browser and allow camera access. The demo downloads the MediaPipe runtime and hand-landmarker model from their public CDNs the first time it starts; the camera frames are processed locally and are not uploaded. Your recorded examples are stored in that browser's local storage on that device. They are not shared with the Python app or other browsers. An empty-camera recording is accepted for the special `idle` label; other labels require a detected hand, and live recognition is suppressed when no hand is currently detected. The browser prototype uses a simple nearest-example comparison, so it can produce false positives and needs several varied examples per label. It is for trying the project, not reliable communication.

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

Record each label in several separate runs. Each run is saved as a separate session, which lets training hold out whole sessions for a more honest score. Record each class at least three times, closing and rerunning the command between sessions. Aim for a similar number of examples per label.

```bash
python -m src.collect --label hola --samples 20
python -m src.collect --label gracias --samples 20
python -m src.collect --label idle --samples 20
```

Run each command again in a new session; do not put all examples for a label in one run. For `idle`, record a mix of hands out of view and **hands visible while moving without signing**. Include the kinds of movement that caused false predictions: moving a hand around the frame, casual waving, changing hand shapes, and moving hands into or out of view. Do not fill the idle class only with empty frames; the model needs examples of ordinary hand activity that is not one of the target signs. The prompt reminds you what to capture.

A useful starting dataset is 3–5 sessions per label, with 20–30 examples per session. If the model predicts `hola` during ordinary movement, record more `idle` examples that look like that movement. If it misses `gracias`, record more varied `gracias` examples. The collector appends new sessions to the current dataset, so you can add examples without starting over. Retrain after adding data.

The dataset is stored in `data/signs.npz`; the trained model is `models/sign_classifier.pkl`. Both stay on your computer and are ignored by Git. Webcam-derived landmarks can still be personal data, so share them only intentionally.

## Train and recognize

```bash
python -m src.train
python -m src.infer
```

The recognizer runs MediaPipe on CPU for predictable cross-platform behavior. It combines motion with a wrist-centered, scale-normalized hand-shape description. It samples camera frames onto a fixed one-second timeline, so slower and faster camera frame rates use the same feature window. It requires several consistent predictions before showing a sign and holds the result briefly. The `idle` class can suppress false activations when its examples cover ordinary movement well.

If ordinary movement still triggers a sign, try a higher confidence threshold:

```bash
python -m src.infer --threshold 0.8
```

The default is `0.60`. A higher threshold can reduce false activations, but may also cause real signs to be missed. Watch the on-screen candidate and confidence while testing, and add representative `idle` examples before relying on threshold changes alone.

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
- **A sign appears during ordinary movement:** record that type of movement as `idle` in multiple sessions, retrain, and try a higher `--threshold` such as `0.8`.
- **A real sign is missed:** add more varied examples for that sign, check that the hand is well lit and visible, and only then try lowering `--threshold` slightly.
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
