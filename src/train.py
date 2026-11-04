"""Train and save the sign classifier from recorded motion windows."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.utils import FEATURE_COUNT, FEATURE_SCHEMA_VERSION, FRAME_FEATURES, WINDOW_SIZE, extract_kinematic_features


def load_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Record samples with src.collect first.")
    with np.load(path, allow_pickle=False) as data:
        if not {"X", "y"}.issubset(data.files):
            raise ValueError("Dataset must contain arrays named X and y.")
        if "feature_version" not in data.files or int(data["feature_version"]) != FEATURE_SCHEMA_VERSION:
            raise ValueError("Dataset uses an older format. Back it up and record fresh samples with src.collect.")
        if "session_id" not in data.files:
            raise ValueError("Dataset is missing recording session IDs. Record fresh samples with src.collect.")
        raw, labels, sessions = data["X"], data["y"].astype(str), data["session_id"].astype(str)

    expected = WINDOW_SIZE * FRAME_FEATURES
    if raw.ndim != 2 or raw.shape[1] != expected:
        raise ValueError(f"Expected X to have shape (samples, {expected}), got {raw.shape}.")
    if len(raw) != len(labels) or len(sessions) != len(labels) or len(labels) < 2:
        raise ValueError("Dataset needs at least two labeled samples with matching session IDs.")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Record at least two different labels before training.")
    if np.any(counts < 2):
        missing = ", ".join(f"{name} ({count})" for name, count in zip(classes, counts) if count < 2)
        raise ValueError(f"Each class needs at least 2 samples for a train/test split: {missing}.")

    features = np.stack([
        extract_kinematic_features(sample.reshape(WINDOW_SIZE, FRAME_FEATURES))
        for sample in raw
    ])
    return features, labels, sessions


def session_holdout(features, labels, sessions):
    """Hold out entire recording sessions, if every class has multiple sessions."""
    rng = np.random.default_rng(42)
    test_indices = []
    for label in np.unique(labels):
        class_sessions = np.unique(sessions[labels == label])
        if len(class_sessions) < 2:
            return None
        held_out = rng.choice(class_sessions)
        test_indices.extend(np.flatnonzero((labels == label) & (sessions == held_out)))
    test_indices = np.asarray(sorted(test_indices), dtype=int)
    train_mask = np.ones(len(labels), dtype=bool)
    train_mask[test_indices] = False
    train_indices = np.flatnonzero(train_mask)
    if len(np.unique(labels[train_indices])) != len(np.unique(labels)):
        return None
    return train_indices, test_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/signs.npz"), help="Recorded dataset")
    parser.add_argument("--output", type=Path, default=Path("models/sign_classifier.pkl"), help="Model output path")
    parser.add_argument("--trees", type=int, default=200, help="Number of random forest trees")
    args = parser.parse_args()
    if args.trees < 1:
        parser.error("--trees must be at least 1")

    try:
        features, labels, sessions = load_dataset(args.data)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    classes, counts = np.unique(labels, return_counts=True)
    print(f"Loaded {len(labels)} samples across {len(classes)} classes.")
    for name, count in zip(classes, counts):
        print(f"  {name}: {count}")

    split = session_holdout(features, labels, sessions)
    if split is not None:
        train_indices, test_indices = split
        x_train, x_test = features[train_indices], features[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
    else:
        x_train = x_test = y_train = y_test = None

    model = RandomForestClassifier(n_estimators=args.trees, class_weight="balanced", random_state=42)
    if x_train is not None:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        print(f"Holdout accuracy: {accuracy_score(y_test, predictions):.2%}")
        print(classification_report(y_test, predictions, zero_division=0))
        print("Holdout is separated by recording session; it still measures only this signer and setup.")
    else:
        print("No session holdout: record each class in at least two separate sessions for evaluation.")
    model.fit(features, labels)
    model._sign_tracker_feature_version = FEATURE_SCHEMA_VERSION

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
