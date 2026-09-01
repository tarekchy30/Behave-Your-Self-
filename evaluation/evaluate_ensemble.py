import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from models.ensemble import BehavioralEnsemble


# ============================================================
# PATHS
# ============================================================
TRAIN_OWNER = os.path.join(
    PROJECT_ROOT,
    "data",
    "train",
    "owner"
)

TEST_OWNER = os.path.join(
    PROJECT_ROOT,
    "data",
    "test",
    "owner"
)

TEST_IMPOSTOR = os.path.join(
    PROJECT_ROOT,
    "data",
    "test",
    "impostor"
)




MODEL_PATH = os.path.join(
    PROJECT_ROOT, "trained_model", "ensemble_model.pth"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT, "trained_model", "ensemble_scaler.pkl"
)


FEATURES = [
    "key_dwell_mean",
    "key_dwell_std",
    "key_dwell_median",
    "flight_time_mean",
    "flight_time_std",
    "flight_time_median",
    "typing_event_rate",
    "mouse_velocity_mean",
    "mouse_velocity_std",
    "mouse_distance_mean",
    "mouse_direction_change",
    "click_interval_mean",
    "click_interval_std"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_folder(folder):

    data = []

    if not os.path.exists(folder):

        raise FileNotFoundError(
            f"Folder does not exist: {folder}"
        )

    for file in sorted(os.listdir(folder)):

        if file.lower().endswith("_features.csv"):

            path = os.path.join(
                folder,
                file
            )

            df = pd.read_csv(path)

            if df.empty:
                continue

            missing = [
                feature
                for feature in FEATURES
                if feature not in df.columns
            ]

            if missing:

                raise RuntimeError(
                    f"Missing features in {path}: {missing}"
                )

            x = df[FEATURES].copy()

            x = x.replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

            data.append(
                x.values.astype(
                    np.float32
                )
            )

    if not data:

        raise RuntimeError(
            f"No feature files found: {folder}"
        )

    return np.vstack(data)


# ============================================================
# EMBEDDINGS
# ============================================================

def embeddings(model, device, data):

    result = []

    with torch.no_grad():

        for row in data:

            x = torch.tensor(
                row,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0).unsqueeze(1)

            emb, _ = model(x)

            result.append(
                emb.squeeze(0)
                .cpu()
                .numpy()
            )

    return np.array(result)


# ============================================================
# MAIN
# ============================================================

def evaluate():

    print("=" * 60)
    print("ENSEMBLE MODEL TESTING")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model = BehavioralEnsemble()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # -------------------------------
    # LOAD
    # -------------------------------

    train_owner = load_folder(
        TRAIN_OWNER
    )

    test_owner = load_folder(
        TEST_OWNER
    )

    test_impostor = load_folder(
        TEST_IMPOSTOR
    )

    print()
    print("Training owner samples :", len(train_owner))
    print("Test owner samples     :", len(test_owner))
    print("Test impostor samples  :", len(test_impostor))

    # -------------------------------
    # SCALE
    # -------------------------------

    train_owner = (
        train_owner - scaler["mean"]
    ) / scaler["std"]

    test_owner = (
        test_owner - scaler["mean"]
    ) / scaler["std"]

    test_impostor = (
        test_impostor - scaler["mean"]
    ) / scaler["std"]

    # -------------------------------
    # EMBEDDINGS
    # -------------------------------

    owner_profile = np.mean(
        embeddings(
            model,
            device,
            train_owner
        ),
        axis=0
    )

    owner_test = embeddings(
        model,
        device,
        test_owner
    )

    impostor_test = embeddings(
        model,
        device,
        test_impostor
    )

    # -------------------------------
    # DISTANCES
    # -------------------------------

    owner_dist = np.linalg.norm(
        owner_test - owner_profile,
        axis=1
    )

    impostor_dist = np.linalg.norm(
        impostor_test - owner_profile,
        axis=1
    )

    # -------------------------------
    # THRESHOLD
    # -------------------------------

    threshold = (
        np.max(owner_dist)
        + np.min(impostor_dist)
    ) / 2

    owner_accept = owner_dist <= threshold
    impostor_reject = impostor_dist > threshold

    accuracy = (
        owner_accept.sum()
        + impostor_reject.sum()
    ) / (
        len(owner_dist)
        + len(impostor_dist)
    )

    far = (
        (~impostor_reject).sum()
        / len(impostor_dist)
    )

    frr = (
        (~owner_accept).sum()
        / len(owner_dist)
    )

    # -------------------------------
    # RESULTS
    # -------------------------------

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"Owner distance    : {owner_dist.mean():.4f}")
    print(f"Impostor distance : {impostor_dist.mean():.4f}")
    print(f"Threshold         : {threshold:.4f}")

    print()
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"FAR      : {far * 100:.2f}%")
    print(f"FRR      : {frr * 100:.2f}%")

    print()
    print("=" * 60)
    print("AUTHENTICATION EXAMPLES")
    print("=" * 60)

    for i, d in enumerate(owner_dist[:5], 1):
        result = "OWNER" if d <= threshold else "IMPOSTOR"
        print(
            f"Owner test {i:02d}: "
            f"{d:.4f} -> {result}"
        )

    for i, d in enumerate(impostor_dist[:5], 1):
        result = "OWNER" if d <= threshold else "IMPOSTOR"
        print(
            f"Impostor test {i:02d}: "
            f"{d:.4f} -> {result}"
        )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()