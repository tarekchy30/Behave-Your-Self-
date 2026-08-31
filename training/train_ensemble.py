import os
import sys
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Allow imports from project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from models.ensemble import BehavioralEnsemble


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_OWNER = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "owner"
)

PROCESSED_IMPOSTOR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "impostor"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "trained_model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ensemble_model.pth"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "ensemble_scaler.pkl"
)

SEQUENCE_LENGTH = 1
INPUT_FEATURES = 13

EMBEDDING_SIZE = 32

EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.001

MARGIN = 1.0

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("Training device:", DEVICE)


# ============================================================
# FEATURE COLUMNS
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
# LOAD PROCESSED FILES
# ============================================================

def load_processed_files(folder):

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
# READ FEATURE VECTOR
# ============================================================

def read_feature_file(path):

    df = pd.read_csv(path)

    if df.empty:
        return []

    # Make sure all expected columns exist

    missing = [
        c for c in FEATURE_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing features in {path}: {missing}"
        )

    values = df[FEATURE_COLUMNS].copy()

    values = values.replace(
        [np.inf, -np.inf],
        np.nan
    )

    values = values.fillna(0.0)

    return values.values.astype(
        np.float32
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    owner_files = load_processed_files(
        PROCESSED_OWNER
    )

    impostor_files = load_processed_files(
        PROCESSED_IMPOSTOR
    )

    print()
    print("=" * 60)
    print("DATASET")
    print("=" * 60)

    print(
        f"Owner files    : {len(owner_files)}"
    )

    print(
        f"Impostor files : {len(impostor_files)}"
    )

    owner_samples = []

    for path in owner_files:

        features = read_feature_file(path)

        for row in features:

            owner_samples.append(row)

    impostor_samples = []

    for path in impostor_files:

        features = read_feature_file(path)

        for row in features:

            impostor_samples.append(row)

    print(
        f"Owner windows    : {len(owner_samples)}"
    )

    print(
        f"Impostor windows : {len(impostor_samples)}"
    )

    if len(owner_samples) == 0:
        raise RuntimeError(
            "No owner feature data found."
        )

    if len(impostor_samples) == 0:
        raise RuntimeError(
            "No impostor feature data found."
        )

    return (
        np.array(owner_samples),
        np.array(impostor_samples)
    )


# ============================================================
# NORMALIZATION
# ============================================================

def fit_scaler(owner, impostor):

    all_data = np.vstack(
        [owner, impostor]
    )

    mean = np.mean(
        all_data,
        axis=0
    )

    std = np.std(
        all_data,
        axis=0
    )

    # Prevent division by zero

    std[std < 1e-8] = 1.0

    scaler = {
        "mean": mean,
        "std": std
    }

    return scaler


def transform(data, scaler):

    return (
        data - scaler["mean"]
    ) / scaler["std"]


# ============================================================
# BUILD POSITIVE / NEGATIVE PAIRS
# ============================================================

def create_pairs(owner, impostor):

    positive_pairs = []
    negative_pairs = []

    # --------------------------------------------------------
    # Positive pairs
    # --------------------------------------------------------

    # Genuine vs genuine

    if len(owner) >= 2:

        for i in range(len(owner)):

            for j in range(i + 1, len(owner)):

                positive_pairs.append(
                    (
                        owner[i],
                        owner[j],
                        1
                    )
                )

    # --------------------------------------------------------
    # Negative pairs
    # --------------------------------------------------------

    # Genuine vs impostor

    for owner_sample in owner:

        for impostor_sample in impostor:

            negative_pairs.append(
                (
                    owner_sample,
                    impostor_sample,
                    0
                )
            )

    # --------------------------------------------------------
    # Balance dataset
    # --------------------------------------------------------

    if len(positive_pairs) == 0:

        # With only one owner window we cannot
        # construct a real positive pair.

        # Create a self-positive pair ONLY so the
        # proof-of-concept model can execute.

        positive_pairs.append(
            (
                owner[0],
                owner[0],
                1
            )
        )

    number_positive = len(
        positive_pairs
    )

    if len(negative_pairs) > number_positive:

        negative_pairs = random.sample(
            negative_pairs,
            number_positive
        )

    pairs = (
        positive_pairs +
        negative_pairs
    )

    random.shuffle(pairs)

    print()
    print(
        f"Positive pairs : {len(positive_pairs)}"
    )

    print(
        f"Negative pairs : {len(negative_pairs)}"
    )

    print(
        f"Total pairs    : {len(pairs)}"
    )

    return pairs


# ============================================================
# DATASET CLASS
# ============================================================

class PairDataset(Dataset):

    def __init__(self, pairs):

        self.pairs = pairs

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, index):

        a, b, label = self.pairs[index]

        # Add sequence dimension

        a = torch.tensor(
            a,
            dtype=torch.float32
        ).unsqueeze(0)

        b = torch.tensor(
            b,
            dtype=torch.float32
        ).unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return a, b, label


# ============================================================
# CONTRASTIVE LOSS
# ============================================================

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):

        super().__init__()

        self.margin = margin

    def forward(
        self,
        distance,
        label
    ):

        positive_loss = (
            label *
            torch.pow(distance, 2)
        )

        negative_loss = (
            (1 - label) *
            torch.pow(
                torch.clamp(
                    self.margin - distance,
                    min=0.0
                ),
                2
            )
        )

        loss = (
            positive_loss +
            negative_loss
        ).mean()

        return loss


# ============================================================
# TRAINING
# ============================================================

def train():

    owner, impostor = load_dataset()

    # --------------------------------------------------------
    # Fit scaler
    # --------------------------------------------------------

    scaler = fit_scaler(
        owner,
        impostor
    )

    owner = transform(
        owner,
        scaler
    )

    impostor = transform(
        impostor,
        scaler
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    with open(
        SCALER_PATH,
        "wb"
    ) as f:

        pickle.dump(
            scaler,
            f
        )

    print()
    print(
        f"Scaler saved: {SCALER_PATH}"
    )

    # --------------------------------------------------------
    # Create pairs
    # --------------------------------------------------------

    pairs = create_pairs(
        owner,
        impostor
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = PairDataset(
        pairs
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = BehavioralEnsemble()

    model = model.to(DEVICE)

    print()
    print(
        "Model:"
    )

    print(model)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = ContrastiveLoss(
        margin=MARGIN
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING ENSEMBLE")
    print("=" * 60)

    best_loss = float("inf")

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0

        for a, b, label in loader:

            a = a.to(DEVICE)

            b = b.to(DEVICE)

            label = label.to(DEVICE)

            optimizer.zero_grad()

            embedding_a, info_a = model(a)

            embedding_b, info_b = model(b)

            distance = F.pairwise_distance(
                embedding_a,
                embedding_b
            )

            loss = criterion(
                distance,
                label
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

        average_loss = (
            total_loss /
            max(len(loader), 1)
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} "
            f"| Loss: {average_loss:.6f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if average_loss < best_loss:

            best_loss = average_loss

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_features":
                        INPUT_FEATURES,

                    "embedding_size":
                        EMBEDDING_SIZE,

                    "feature_columns":
                        FEATURE_COLUMNS,

                    "ensemble_weights":
                        model.ensemble_weights
                        .detach()
                        .cpu()
                        .numpy()
                },
                MODEL_PATH
            )

            print(
                "  ✓ Best ensemble model saved"
            )

    print()
    print("=" * 60)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Model : {MODEL_PATH}"
    )

    print(
        f"Scaler: {SCALER_PATH}"
    )

    print()
    print(
        "Learned ensemble weights:"
    )

    weights = torch.softmax(
        model.ensemble_weights,
        dim=0
    )

    print(
        weights.detach().cpu()
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()