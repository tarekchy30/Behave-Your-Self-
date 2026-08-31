import os
import sys
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from preprocessing.features import create_windows


RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw"
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)


def process_file(input_path, output_path):

    print("-" * 60)
    print("Processing:", input_path)

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

        print(
            "Windows created:",
            len(features)
        )

        print(
            "Saved:",
            output_path
        )

        return True

    except Exception as e:

        print(
            "ERROR:",
            e
        )

        return False


def process_dataset():

    print("=" * 60)
    print("PROCESSING BEHAVIOR DATASET")
    print("=" * 60)

    owner_processed = 0
    impostor_processed = 0
    failed = 0

    # --------------------------------------------------------
    # OWNER
    # --------------------------------------------------------

    owner_dir = os.path.join(
        RAW_DIR,
        "owner"
    )

    if os.path.exists(owner_dir):

        for filename in os.listdir(owner_dir):

            if not filename.lower().endswith(".csv"):
                continue

            input_path = os.path.join(
                owner_dir,
                filename
            )

            output_name = (
                os.path.splitext(filename)[0]
                + "_features.csv"
            )

            output_path = os.path.join(
                PROCESSED_DIR,
                "owner",
                output_name
            )

            if process_file(
                input_path,
                output_path
            ):

                owner_processed += 1

            else:

                failed += 1

    # --------------------------------------------------------
    # IMPOSTORS
    # --------------------------------------------------------

    impostor_dir = os.path.join(
        RAW_DIR,
        "impostor"
    )

    if os.path.exists(impostor_dir):

        for filename in os.listdir(impostor_dir):

            if not filename.lower().endswith(".csv"):
                continue

            input_path = os.path.join(
                impostor_dir,
                filename
            )

            output_name = (
                os.path.splitext(filename)[0]
                + "_features.csv"
            )

            output_path = os.path.join(
                PROCESSED_DIR,
                "impostor",
                output_name
            )

            if process_file(
                input_path,
                output_path
            ):

                impostor_processed += 1

            else:

                failed += 1

    print()
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    print(
        "Owner files    :",
        owner_processed
    )

    print(
        "Impostor files :",
        impostor_processed
    )

    print(
        "Failed files   :",
        failed
    )


if __name__ == "__main__":

    process_dataset()