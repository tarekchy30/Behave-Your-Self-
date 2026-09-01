import os
import sys
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ============================================================
# PROJECT ROOT
# ============================================================

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
    PROJECT_ROOT,
    "trained_model",
    "ensemble_model.pth"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_model",
    "ensemble_scaler.pkl"
)

PLOTS_DIR = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "plots"
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


# ============================================================
# FEATURES
# ============================================================

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

        if not file.lower().endswith(
            "_features.csv"
        ):
            continue

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
        )

        x = x.fillna(0)

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

def embeddings(
    model,
    device,
    data
):

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
# SAVE EVALUATION METRICS GRAPH
# ============================================================

def save_metrics_graph(
    accuracy,
    far,
    frr
):

    metrics = {
        "Accuracy": accuracy * 100,
        "FAR": far * 100,
        "FRR": frr * 100
    }

    plt.figure(
        figsize=(9, 6)
    )

    bars = plt.bar(
        metrics.keys(),
        metrics.values()
    )

    plt.ylabel(
        "Percentage (%)"
    )

    plt.xlabel(
        "Evaluation Metric"
    )

    plt.title(
        "Behavioral Authentication Performance"
    )

    plt.ylim(
        0,
        100
    )

    # Values above bars

    for bar in bars:

        value = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 1,

            f"{value:.2f}%",

            ha="center",

            va="bottom"
        )

    plt.tight_layout()

    path = os.path.join(
        PLOTS_DIR,
        "evaluation_metrics.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Metrics graph saved: {path}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    owner_accept,
    impostor_reject
):

    true_owner_pred_owner = np.sum(
        owner_accept
    )

    true_owner_pred_impostor = (
        len(owner_accept)
        - true_owner_pred_owner
    )

    true_impostor_pred_owner = (
        len(impostor_reject)
        - np.sum(impostor_reject)
    )

    true_impostor_pred_impostor = np.sum(
        impostor_reject
    )

    cm = np.array([
        [
            true_owner_pred_owner,
            true_owner_pred_impostor
        ],
        [
            true_impostor_pred_owner,
            true_impostor_pred_impostor
        ]
    ])

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "Authentication Confusion Matrix"
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        [
            "Predicted Owner",
            "Predicted Impostor"
        ]
    )

    plt.yticks(
        [0, 1],
        [
            "Actual Owner",
            "Actual Impostor"
        ]
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.tight_layout()

    path = os.path.join(
        PLOTS_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix saved: {path}"
    )


# ============================================================
# DISTANCE DISTRIBUTION
# ============================================================

def save_distance_distribution(
    owner_dist,
    impostor_dist,
    threshold
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        owner_dist,
        bins=20,
        alpha=0.7,
        label="Owner"
    )

    plt.hist(
        impostor_dist,
        bins=20,
        alpha=0.7,
        label="Impostor"
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.4f}"
    )

    plt.xlabel(
        "Embedding Distance"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Owner vs Impostor Embedding Distance"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        PLOTS_DIR,
        "distance_distribution.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Distance distribution saved: {path}"
    )


# ============================================================
# ROC CURVE
# ============================================================

def save_roc_curve(
    owner_dist,
    impostor_dist
):

    all_distances = np.concatenate([
        owner_dist,
        impostor_dist
    ])

    all_labels = np.concatenate([
        np.ones(len(owner_dist)),
        np.zeros(len(impostor_dist))
    ])

    thresholds = np.linspace(
        all_distances.min(),
        all_distances.max(),
        200
    )

    far_values = []
    tar_values = []

    for threshold in thresholds:

        accepted = (
            all_distances <= threshold
        )

        false_accepts = np.sum(
            (all_labels == 0)
            & accepted
        )

        true_accepts = np.sum(
            (all_labels == 1)
            & accepted
        )

        impostor_count = np.sum(
            all_labels == 0
        )

        owner_count = np.sum(
            all_labels == 1
        )

        far_value = (
            false_accepts
            / max(impostor_count, 1)
        )

        tar_value = (
            true_accepts
            / max(owner_count, 1)
        )

        far_values.append(
            far_value
        )

        tar_values.append(
            tar_value
        )

    far_values = np.array(
        far_values
    )

    tar_values = np.array(
        tar_values
    )

    order = np.argsort(
        far_values
    )

    far_values = far_values[
        order
    ]

    tar_values = tar_values[
        order
    ]

    auc = np.trapezoid(
        tar_values,
        far_values
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.plot(
        far_values,
        tar_values,
        linewidth=2,
        label=f"AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "False Acceptance Rate (FAR)"
    )

    plt.ylabel(
        "True Acceptance Rate (TAR)"
    )

    plt.title(
        "ROC Curve - Behavioral Authentication"
    )

    plt.xlim(
        0,
        1
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        PLOTS_DIR,
        "roc_curve.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"ROC curve saved: {path}"
    )

    print(
        f"ROC AUC: {auc:.4f}"
    )

    return auc


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate():

    print("=" * 60)
    print("ENSEMBLE MODEL TESTING")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Evaluation device:",
        device
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model = BehavioralEnsemble()

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(device)

    model.eval()

    print(
        "Model loaded successfully."
    )

    # ========================================================
    # LOAD SCALER
    # ========================================================

    with open(
        SCALER_PATH,
        "rb"
    ) as f:

        scaler = pickle.load(f)

    print(
        "Scaler loaded successfully."
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

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
    print(
        "Training owner samples :",
        len(train_owner)
    )

    print(
        "Test owner samples     :",
        len(test_owner)
    )

    print(
        "Test impostor samples  :",
        len(test_impostor)
    )

    # ========================================================
    # SCALE DATA
    # ========================================================

    train_owner = (
        train_owner
        - scaler["mean"]
    ) / scaler["std"]

    test_owner = (
        test_owner
        - scaler["mean"]
    ) / scaler["std"]

    test_impostor = (
        test_impostor
        - scaler["mean"]
    ) / scaler["std"]

    # ========================================================
    # GENERATE EMBEDDINGS
    # ========================================================

    print()
    print(
        "Generating embeddings..."
    )

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

    # ========================================================
    # DISTANCES
    # ========================================================

    owner_dist = np.linalg.norm(
        owner_test
        - owner_profile,
        axis=1
    )

    impostor_dist = np.linalg.norm(
        impostor_test
        - owner_profile,
        axis=1
    )

    # ========================================================
    # THRESHOLD
    # ========================================================

    threshold = (
        np.max(owner_dist)
        + np.min(impostor_dist)
    ) / 2

    owner_accept = (
        owner_dist <= threshold
    )

    impostor_reject = (
        impostor_dist > threshold
    )

    # ========================================================
    # METRICS
    # ========================================================

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

    # ========================================================
    # SAVE GRAPHS
    # ========================================================

    print()
    print("=" * 60)
    print("GENERATING EVALUATION GRAPHS")
    print("=" * 60)

    save_metrics_graph(
        accuracy,
        far,
        frr
    )

    save_confusion_matrix(
        owner_accept,
        impostor_reject
    )

    save_distance_distribution(
        owner_dist,
        impostor_dist,
        threshold
    )

    auc = save_roc_curve(
        owner_dist,
        impostor_dist
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"Owner distance    : "
        f"{owner_dist.mean():.4f}"
    )

    print(
        f"Impostor distance : "
        f"{impostor_dist.mean():.4f}"
    )

    print(
        f"Threshold         : "
        f"{threshold:.4f}"
    )

    print()

    print(
        f"Accuracy : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"FAR      : "
        f"{far * 100:.2f}%"
    )

    print(
        f"FRR      : "
        f"{frr * 100:.2f}%"
    )

    print(
        f"ROC AUC  : "
        f"{auc:.4f}"
    )

    # ========================================================
    # AUTHENTICATION EXAMPLES
    # ========================================================

    print()
    print("=" * 60)
    print("AUTHENTICATION EXAMPLES")
    print("=" * 60)

    for i, d in enumerate(
        owner_dist[:5],
        1
    ):

        result = (
            "OWNER"
            if d <= threshold
            else "IMPOSTOR"
        )

        print(
            f"Owner test {i:02d}: "
            f"{d:.4f} -> {result}"
        )

    for i, d in enumerate(
        impostor_dist[:5],
        1
    ):

        result = (
            "OWNER"
            if d <= threshold
            else "IMPOSTOR"
        )

        print(
            f"Impostor test {i:02d}: "
            f"{d:.4f} -> {result}"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print()
    print(
        "Graphs saved in:"
    )

    print(
        PLOTS_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    evaluate()