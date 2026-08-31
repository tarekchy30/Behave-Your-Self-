import os
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import random

import joblib
import numpy as np
import pandas as pd
import torch



from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

from models.siamese import (
    SiameseNetwork,
    ContrastiveLoss,
    euclidean_distance
)


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DIR = "data/processed"

MODEL_DIR = "trained_model"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "siamese_model.pth"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

INPUT_SIZE = 13

EMBEDDING_SIZE = 32

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 0.001

MARGIN = 1.0

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)


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

def load_processed_data():

    owner_files = []

    impostor_files = []

    for root, dirs, files in os.walk(
        PROCESSED_DIR
    ):

        for file in files:

            if not file.endswith(
                "_features.csv"
            ):
                continue

            path = os.path.join(
                root,
                file
            )

            # ------------------------------------------------
            # Determine user type from filename/path
            # ------------------------------------------------

            if "impostor" in path.lower():

                impostor_files.append(
                    path
                )

            else:

                owner_files.append(
                    path
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

    if len(owner_files) == 0:

        raise RuntimeError(
            "No owner feature files found."
        )

    if len(impostor_files) == 0:

        raise RuntimeError(
            "No impostor feature files found."
        )

    return owner_files, impostor_files


# ============================================================
# LOAD ONE CSV
# ============================================================

def load_features(path):

    df = pd.read_csv(path)

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing features in {path}: "
            f"{missing}"
        )

    X = df[
        FEATURE_COLUMNS
    ].copy()

    # Convert everything to numeric
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace invalid values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values
    X = X.fillna(0.0)

    return X.values.astype(
        np.float32
    )


# ============================================================
# LOAD ALL SAMPLES
# ============================================================

def load_all_samples(
    owner_files,
    impostor_files
):

    owner_samples = []

    impostor_samples = []

    # --------------------------------------------------------
    # Owner
    # --------------------------------------------------------

    for path in owner_files:

        X = load_features(path)

        for sample in X:

            owner_samples.append(
                sample
            )

    # --------------------------------------------------------
    # Impostor
    # --------------------------------------------------------

    for path in impostor_files:

        X = load_features(path)

        for sample in X:

            impostor_samples.append(
                sample
            )

    owner_samples = np.array(
        owner_samples,
        dtype=np.float32
    )

    impostor_samples = np.array(
        impostor_samples,
        dtype=np.float32
    )

    print()
    print(
        f"Owner windows    : "
        f"{len(owner_samples)}"
    )

    print(
        f"Impostor windows : "
        f"{len(impostor_samples)}"
    )

    return (
        owner_samples,
        impostor_samples
    )


# ============================================================
# CREATE SIAMESE PAIRS
# ============================================================

def create_pairs(
    owner_samples,
    impostor_samples
):

    pairs_a = []

    pairs_b = []

    labels = []

    # --------------------------------------------------------
    # POSITIVE PAIRS
    # --------------------------------------------------------
    # Owner + Owner
    # Label = 1
    # --------------------------------------------------------

    num_positive = min(
        len(owner_samples) * 2,
        5000
    )

    for _ in range(
        num_positive
    ):

        index_a = random.randrange(
            len(owner_samples)
        )

        index_b = random.randrange(
            len(owner_samples)
        )

        # Avoid pairing a sample with itself
        if len(owner_samples) > 1:

            while index_b == index_a:

                index_b = random.randrange(
                    len(owner_samples)
                )

        pairs_a.append(
            owner_samples[index_a]
        )

        pairs_b.append(
            owner_samples[index_b]
        )

        labels.append(1.0)

    # --------------------------------------------------------
    # NEGATIVE PAIRS
    # --------------------------------------------------------
    # Owner + Impostor
    # Label = 0
    # --------------------------------------------------------

    num_negative = min(
        len(owner_samples) * 2,
        5000
    )

    for _ in range(
        num_negative
    ):

        owner_index = random.randrange(
            len(owner_samples)
        )

        impostor_index = random.randrange(
            len(impostor_samples)
        )

        pairs_a.append(
            owner_samples[
                owner_index
            ]
        )

        pairs_b.append(
            impostor_samples[
                impostor_index
            ]
        )

        labels.append(0.0)

    pairs_a = np.array(
        pairs_a,
        dtype=np.float32
    )

    pairs_b = np.array(
        pairs_b,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    indices = np.arange(
        len(labels)
    )

    np.random.shuffle(
        indices
    )

    pairs_a = pairs_a[
        indices
    ]

    pairs_b = pairs_b[
        indices
    ]

    labels = labels[
        indices
    ]

    print()
    print(
        f"Total pairs: {len(labels)}"
    )

    print(
        f"Positive pairs: "
        f"{np.sum(labels == 1)}"
    )

    print(
        f"Negative pairs: "
        f"{np.sum(labels == 0)}"
    )

    return (
        pairs_a,
        pairs_b,
        labels
    )


# ============================================================
# PYTORCH DATASET
# ============================================================

class PairDataset(Dataset):

    def __init__(
        self,
        pairs_a,
        pairs_b,
        labels
    ):

        self.a = torch.tensor(
            pairs_a,
            dtype=torch.float32
        )

        self.b = torch.tensor(
            pairs_b,
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.float32
        )


    def __len__(self):

        return len(
            self.labels
        )


    def __getitem__(self, index):

        return (
            self.a[index],
            self.b[index],
            self.labels[index]
        )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    model,
    train_loader,
    validation_loader,
    device
):

    criterion = ContrastiveLoss(
        margin=MARGIN
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_validation_loss = float(
        "inf"
    )

    print()
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(
        EPOCHS
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        total_train_loss = 0.0

        for (
            sample_a,
            sample_b,
            labels
        ) in train_loader:

            sample_a = sample_a.to(
                device
            )

            sample_b = sample_b.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            embedding_a, embedding_b = (
                model(
                    sample_a,
                    sample_b
                )
            )

            loss = criterion(
                embedding_a,
                embedding_b,
                labels
            )

            loss.backward()

            optimizer.step()

            total_train_loss += (
                loss.item()
            )

        train_loss = (
            total_train_loss
            /
            len(train_loader)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        total_validation_loss = 0.0

        with torch.no_grad():

            for (
                sample_a,
                sample_b,
                labels
            ) in validation_loader:

                sample_a = sample_a.to(
                    device
                )

                sample_b = sample_b.to(
                    device
                )

                labels = labels.to(
                    device
                )

                embedding_a, embedding_b = (
                    model(
                        sample_a,
                        sample_b
                    )
                )

                loss = criterion(
                    embedding_a,
                    embedding_b,
                    labels
                )

                total_validation_loss += (
                    loss.item()
                )

        validation_loss = (
            total_validation_loss
            /
            len(validation_loader)
        )

        print(
            f"Epoch "
            f"{epoch + 1:03d}/{EPOCHS} "
            f"| Train Loss: "
            f"{train_loss:.6f} "
            f"| Validation Loss: "
            f"{validation_loss:.6f}"
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "  ✓ Best model saved"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"Training device: {device}"
    )

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    (
        owner_files,
        impostor_files
    ) = load_processed_data()

    # --------------------------------------------------------
    # Load samples
    # --------------------------------------------------------

    (
        owner_samples,
        impostor_samples
    ) = load_all_samples(
        owner_files,
        impostor_files
    )

    # --------------------------------------------------------
    # Create pairs
    # --------------------------------------------------------

    (
        pairs_a,
        pairs_b,
        labels
    ) = create_pairs(
        owner_samples,
        impostor_samples
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    split = int(
        len(labels) * 0.8
    )

    train_a = pairs_a[
        :split
    ]

    train_b = pairs_b[
        :split
    ]

    train_labels = labels[
        :split
    ]

    validation_a = pairs_a[
        split:
    ]

    validation_b = pairs_b[
        split:
    ]

    validation_labels = labels[
        split:
    ]

    # --------------------------------------------------------
    # Feature scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    # Fit only on training data
    combined_train = np.vstack([
        train_a,
        train_b
    ])

    scaler.fit(
        combined_train
    )

    train_a = scaler.transform(
        train_a
    ).astype(
        np.float32
    )

    train_b = scaler.transform(
        train_b
    ).astype(
        np.float32
    )

    validation_a = scaler.transform(
        validation_a
    ).astype(
        np.float32
    )

    validation_b = scaler.transform(
        validation_b
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    print()
    print(
        f"Scaler saved to: "
        f"{SCALER_PATH}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = PairDataset(
        train_a,
        train_b,
        train_labels
    )

    validation_dataset = PairDataset(
        validation_a,
        validation_b,
        validation_labels
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SiameseNetwork(
        input_size=INPUT_SIZE,
        embedding_size=EMBEDDING_SIZE
    )

    model = model.to(
        device
    )

    print()
    print(
        f"Input features: "
        f"{INPUT_SIZE}"
    )

    print(
        f"Embedding size: "
        f"{EMBEDDING_SIZE}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_model(
        model,
        train_loader,
        validation_loader,
        device
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Scaler: {SCALER_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()