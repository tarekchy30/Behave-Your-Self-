import os
import sys
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from models.ensemble import BehavioralEnsemble


# ============================================================
# PATHS
# ============================================================

OWNER_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "owner"
)

IMPOSTOR_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "impostor"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_model",
    "ensemble_model.pth"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_model",
    "ensemble_scaler.pkl"
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_COLUMNS = [
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
# LOAD FEATURE FILES
# ============================================================

def load_files(folder):

    files = []

    if not os.path.exists(folder):
        return files

    for filename in os.listdir(folder):

        if filename.endswith("_features.csv"):

            files.append(
                os.path.join(
                    folder,
                    filename
                )
            )

    return sorted(files)


# ============================================================
# READ FEATURES
# ============================================================

def read_features(path):

    df = pd.read_csv(path)

    if df.empty:
        return []

    missing = [
        c for c in FEATURE_COLUMNS
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing features in {path}: {missing}"
        )

    values = df[
        FEATURE_COLUMNS
    ].copy()

    values = values.replace(
        [np.inf, -np.inf],
        np.nan
    )

    values = values.fillna(0.0)

    return values.values.astype(
        np.float32
    )


# ============================================================
# LOAD SCALER
# ============================================================

def load_scaler():

    with open(
        SCALER_PATH,
        "rb"
    ) as f:

        scaler = pickle.load(f)

    return scaler


# ============================================================
# APPLY SCALER
# ============================================================

def scale_features(
    features,
    scaler
):

    return (
        features - scaler["mean"]
    ) / scaler["std"]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
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

    return model, device, checkpoint


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    model,
    device,
    features
):

    embeddings = []

    with torch.no_grad():

        for row in features:

            # [13] -> [1, 13]
            x = torch.tensor(
                row,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            # [1, 13] -> [1, 1, 13]
            x = x.unsqueeze(1)

            embedding, _ = model(x)

            embeddings.append(
                embedding.squeeze(0)
                .cpu()
                .numpy()
            )

    return np.array(
        embeddings
    )


# ============================================================
# EUCLIDEAN DISTANCE
# ============================================================

def distance(a, b):

    return float(
        np.linalg.norm(a - b)
    )


# ============================================================
# FIND THRESHOLD
# ============================================================

def find_threshold(
    genuine_distances,
    impostor_distances
):

    all_distances = np.concatenate(
        [
            genuine_distances,
            impostor_distances
        ]
    )

    best_threshold = None
    best_accuracy = -1

    for threshold in all_distances:

        genuine_correct = np.sum(
            genuine_distances <= threshold
        )

        impostor_correct = np.sum(
            impostor_distances > threshold
        )

        total = (
            len(genuine_distances)
            +
            len(impostor_distances)
        )

        accuracy = (
            genuine_correct
            +
            impostor_correct
        ) / total

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_threshold = threshold

    return (
        best_threshold,
        best_accuracy
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate():

    print()
    print("=" * 60)
    print("ENSEMBLE MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        raise RuntimeError(
            "Ensemble model not found."
        )

    if not os.path.exists(
        SCALER_PATH
    ):

        raise RuntimeError(
            "Ensemble scaler not found."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, device, checkpoint = (
        load_model()
    )

    print()
    print(
        "Model loaded successfully."
    )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------

    scaler = load_scaler()

    # --------------------------------------------------------
    # Owner
    # --------------------------------------------------------

    owner_files = load_files(
        OWNER_DIR
    )

    owner_features = []

    for path in owner_files:

        owner_features.extend(
            read_features(path)
        )

    owner_features = np.array(
        owner_features,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Impostor
    # --------------------------------------------------------

    impostor_files = load_files(
        IMPOSTOR_DIR
    )

    impostor_features = []

    for path in impostor_files:

        impostor_features.extend(
            read_features(path)
        )

    impostor_features = np.array(
        impostor_features,
        dtype=np.float32
    )

    print()
    print(
        "Owner samples   :",
        len(owner_features)
    )

    print(
        "Impostor samples:",
        len(impostor_features)
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    owner_features = scale_features(
        owner_features,
        scaler
    )

    impostor_features = scale_features(
        impostor_features,
        scaler
    )

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print()
    print(
        "Generating owner embeddings..."
    )

    owner_embeddings = create_embeddings(
        model,
        device,
        owner_features
    )

    print(
        "Generating impostor embeddings..."
    )

    impostor_embeddings = create_embeddings(
        model,
        device,
        impostor_features
    )

    # --------------------------------------------------------
    # Owner reference profile
    # --------------------------------------------------------

    owner_profile = np.mean(
        owner_embeddings,
        axis=0
    )

    # --------------------------------------------------------
    # Genuine distances
    # --------------------------------------------------------

    genuine_distances = []

    for embedding in owner_embeddings:

        genuine_distances.append(
            distance(
                embedding,
                owner_profile
            )
        )

    genuine_distances = np.array(
        genuine_distances
    )

    # --------------------------------------------------------
    # Impostor distances
    # --------------------------------------------------------

    impostor_distances = []

    for embedding in impostor_embeddings:

        impostor_distances.append(
            distance(
                embedding,
                owner_profile
            )
        )

    impostor_distances = np.array(
        impostor_distances
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DISTANCE ANALYSIS")
    print("=" * 60)

    print()

    print(
        "Owner distance:"
    )

    print(
        f"  Mean   : {np.mean(genuine_distances):.6f}"
    )

    print(
        f"  Std    : {np.std(genuine_distances):.6f}"
    )

    print(
        f"  Min    : {np.min(genuine_distances):.6f}"
    )

    print(
        f"  Max    : {np.max(genuine_distances):.6f}"
    )

    print()

    print(
        "Impostor distance:"
    )

    print(
        f"  Mean   : {np.mean(impostor_distances):.6f}"
    )

    print(
        f"  Std    : {np.std(impostor_distances):.6f}"
    )

    print(
        f"  Min    : {np.min(impostor_distances):.6f}"
    )

    print(
        f"  Max    : {np.max(impostor_distances):.6f}"
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold, accuracy = find_threshold(
        genuine_distances,
        impostor_distances
    )

    print()
    print("=" * 60)
    print("THRESHOLD")
    print("=" * 60)

    print(
        f"Best threshold : {threshold:.6f}"
    )

    print(
        f"Accuracy       : {accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # FAR
    # --------------------------------------------------------

    false_accepts = np.sum(
        impostor_distances <= threshold
    )

    false_rejects = np.sum(
        genuine_distances > threshold
    )

    far = (
        false_accepts /
        max(len(impostor_distances), 1)
    )

    frr = (
        false_rejects /
        max(len(genuine_distances), 1)
    )

    print()
    print("=" * 60)
    print("SECURITY METRICS")
    print("=" * 60)

    print(
        f"False Acceptance Rate : {far * 100:.2f}%"
    )

    print(
        f"False Rejection Rate  : {frr * 100:.2f}%"
    )

    print(
        f"False Accepts         : {false_accepts}"
    )

    print(
        f"False Rejects         : {false_rejects}"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    evaluate()
 