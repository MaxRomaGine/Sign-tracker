# Sign Tracker

A small, webcam based prototype for recognizing a personal vocabulary of hand signs. It uses MediaPipe landmarks and a Random Forest to classify short motion windows.

This is an experiment, not a general sign language translator. It has no built-in vocabulary or model: you record examples for the signs you want to recognize. It has only been explored with a very small vocabulary (including “hola” and “gracias”), so results will depend heavily on the person, camera, lighting, and examples used for training.

## What it does

- Extracts 3D hand landmarks with MediaPipe Tasks and keeps left and right hands in stable feature slots.
- Summarizes a 30-frame window with two landmark displacements and peak frame-to-frame velocity.
- Trains a Random Forest from locally recorded examples and shows a confidence-gated prediction in a live camera window.

## Setup

Python 3.10 or 3.11 is recommended. MediaPipe does not publish compatible packages for every Python version or platform, so use a supported Python version if installation fails.

```bash
git clone https://github.com/MaxRomaGine/Sign-tracker.git
cd Sign-tracker
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p models
curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -o models/hand_landmarker.task
```

The first run may ask your operating system to grant camera access. Camera device `0` is used by default; pass `--camera 1` to choose another camera.

## Record examples

Record each label in multiple short sessions. Vary your position and speed a little, keep the sign in view, and make examples for every sign you want the model to distinguish. Each sample is one second long; press Space to capture it. Press `q` to stop.

```bash
python -m src.collect --label hola --samples 30
python -m src.collect --label gracias --samples 30
```

Collection appends examples to `data/signs.npz`. This file contains webcam-derived landmarks and stays local by default. Do not commit it unless you intend to share those recordings/features. The landmark format changed in this update, so back up any earlier dataset and record fresh samples rather than mixing formats.

## Train and run

```bash
python -m src.train
python -m src.infer
```

Training prints class counts and, when the dataset is large enough, a small stratified holdout report. A random holdout from the same recording sessions is only a rough debugging signal; it does not measure performance on new people or conditions. The saved model goes to `models/sign_classifier.pkl` and is also local by default.

Useful options:

```bash
python -m src.collect --label hola --samples 40 --camera 1
python -m src.train --data data/signs.npz --output models/my_model.pkl --trees 300
python -m src.infer --model models/my_model.pkl --camera 1 --threshold 0.65
```

## Controls and limitations

- Press `q` to exit collection or live inference; press Space to record one sample.
- A missing hand is encoded as zero landmarks. There is no separate “idle” training class; low motion and low confidence are suppressed instead.
- The MediaPipe `hand_landmarker.task` asset must be downloaded once into `models/` before collection, debugging, or inference.
- The classifier uses a fixed 30-frame window, so performance assumes a reasonably steady camera frame rate.
- Dynamic signs that look similar in a short window, two-handed signs, occlusion, and large differences in signing style may be misclassified.
- This project does not translate continuous signing, recognize facial or grammatical markers, or replace instruction from Deaf signers and language experts.

## Project layout

```text
src/
  collect.py    Record labeled webcam windows
  extractor.py  Extract normalized MediaPipe landmarks
  utils.py      Convert landmark windows to motion features
  train.py      Validate data, train and save a classifier
  infer.py      Run confidence-gated live prediction
  debug.py      Inspect the camera feature pipeline
```

## Next steps

To evaluate this beyond a personal prototype, collect more balanced sessions, separate train and test data by recording session (and signer), compare against a temporal model, and measure false activations during ordinary movement. A useful portfolio demo should show both successful predictions and the current failure cases.
