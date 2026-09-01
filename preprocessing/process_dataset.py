import os
import sys
import pandas as pd

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from preprocessing.features import create_windows


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")

TRAIN_OWNER = 4
TRAIN_IMPOSTOR = 20


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(input_path, output_path):

    print("-" * 60)
    print("Processing:", os.path.basename(input_path))

    try:

        features = create_windows(
            input_path,
            window_seconds=30
        )

        if features is None or features.empty:
            print("WARNING: No feature windows generated")
            return False

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        features.to_csv(
            output_path,
            index=False
        )

        print("Windows:", len(features))
        print("Saved  :", output_path)

        return True

    except Exception as e:

        print("ERROR:", e)
        return False


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset():

    print("=" * 60)
    print("BEHAVIOR DATASET PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # GET RAW FILES
    # --------------------------------------------------------

    owner_dir = os.path.join(RAW_DIR, "owner")
    impostor_dir = os.path.join(RAW_DIR, "impostor")

    owner_files = sorted(
        f for f in os.listdir(owner_dir)
        if f.lower().endswith(".csv")
    )

    impostor_files = sorted(
        f for f in os.listdir(impostor_dir)
        if f.lower().endswith(".csv")
    )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    train_owner = owner_files[:TRAIN_OWNER]
    test_owner = owner_files[TRAIN_OWNER:]

    train_impostor = impostor_files[:TRAIN_IMPOSTOR]
    test_impostor = impostor_files[TRAIN_IMPOSTOR:]

    print()
    print("DATASET SPLIT")
    print("-" * 60)

    print(f"Owner    : {len(owner_files)}")
    print(f"Impostor : {len(impostor_files)}")

    print()
    print("TRAIN")
    print(f"Owner    : {len(train_owner)}")
    print(f"Impostor : {len(train_impostor)}")

    print()
    print("TEST")
    print(f"Owner    : {len(test_owner)}")
    print(f"Impostor : {len(test_impostor)}")

    # --------------------------------------------------------
    # PROCESS TRAIN OWNER
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PROCESSING TRAIN OWNER")
    print("=" * 60)

    for filename in train_owner:

        input_path = os.path.join(
            owner_dir,
            filename
        )

        output_name = (
            os.path.splitext(filename)[0]
            + "_features.csv"
        )

        output_path = os.path.join(
            TRAIN_DIR,
            "owner",
            output_name
        )

        process_file(
            input_path,
            output_path
        )

    # --------------------------------------------------------
    # PROCESS TRAIN IMPOSTOR
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PROCESSING TRAIN IMPOSTOR")
    print("=" * 60)

    for filename in train_impostor:

        input_path = os.path.join(
            impostor_dir,
            filename
        )

        output_name = (
            os.path.splitext(filename)[0]
            + "_features.csv"
        )

        output_path = os.path.join(
            TRAIN_DIR,
            "impostor",
            output_name
        )

        process_file(
            input_path,
            output_path
        )

    # --------------------------------------------------------
    # PROCESS TEST OWNER
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PROCESSING TEST OWNER")
    print("=" * 60)

    for filename in test_owner:

        input_path = os.path.join(
            owner_dir,
            filename
        )

        output_name = (
            os.path.splitext(filename)[0]
            + "_features.csv"
        )

        output_path = os.path.join(
            TEST_DIR,
            "owner",
            output_name
        )

        process_file(
            input_path,
            output_path
        )

    # --------------------------------------------------------
    # PROCESS TEST IMPOSTOR
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PROCESSING TEST IMPOSTOR")
    print("=" * 60)

    for filename in test_impostor:

        input_path = os.path.join(
            impostor_dir,
            filename
        )

        output_name = (
            os.path.splitext(filename)[0]
            + "_features.csv"
        )

        output_path = os.path.join(
            TEST_DIR,
            "impostor",
            output_name
        )

        process_file(
            input_path,
            output_path
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print()
    print("Training data:")
    print(f"  Owner    : {len(train_owner)}")
    print(f"  Impostor : {len(train_impostor)}")

    print()
    print("Testing data:")
    print(f"  Owner    : {len(test_owner)}")
    print(f"  Impostor : {len(test_impostor)}")

    print()
    print("Train directory:")
    print(TRAIN_DIR)

    print()
    print("Test directory:")
    print(TEST_DIR)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    process_dataset()