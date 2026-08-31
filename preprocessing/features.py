import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SECONDS = 30


# ============================================================
# LOAD DATA
# ============================================================

def load_data(csv_path):

    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace("*", "", regex=False)
        .str.replace("\\", "", regex=False)
    )

    if df.empty:
        raise ValueError(
            f"No data found in: {csv_path}"
        )

    # Make sure timestamp is numeric
    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    # Remove invalid timestamps
    df = df.dropna(
        subset=["timestamp"]
    )

    # Sort chronologically
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df

# ============================================================
# KEYBOARD FEATURES
# ============================================================

def extract_keyboard_features(df):

    keyboard = df[
        df["event_type"] == "keyboard"
    ].copy()

    features = {
        "key_dwell_mean": 0.0,
        "key_dwell_std": 0.0,
        "key_dwell_median": 0.0,

        "flight_time_mean": 0.0,
        "flight_time_std": 0.0,
        "flight_time_median": 0.0,

        "typing_event_rate": 0.0
    }

    if len(keyboard) < 2:
        return features

    keyboard = keyboard.sort_values(
        "timestamp"
    )

    # --------------------------------------------------------
    # KEY DWELL TIME
    # --------------------------------------------------------

    downs = keyboard[
        keyboard["key_state"] == "down"
    ]

    ups = keyboard[
        keyboard["key_state"] == "up"
    ]

    dwell_times = []

    for _, down in downs.iterrows():

        later_ups = ups[
            ups["timestamp"] >= down["timestamp"]
        ]

        if len(later_ups) == 0:
            continue

        up = later_ups.iloc[0]

        duration = (
            up["timestamp"]
            -
            down["timestamp"]
        )

        # Ignore unrealistic values
        if 0 < duration < 5:

            dwell_times.append(
                duration
            )

    if len(dwell_times) > 0:

        dwell_times = np.array(
            dwell_times
        )

        features["key_dwell_mean"] = float(
            np.mean(dwell_times)
        )

        features["key_dwell_std"] = float(
            np.std(dwell_times)
        )

        features["key_dwell_median"] = float(
            np.median(dwell_times)
        )

    # --------------------------------------------------------
    # FLIGHT TIME
    # --------------------------------------------------------

    press_times = downs[
        "timestamp"
    ].values

    if len(press_times) >= 2:

        flight_times = np.diff(
            press_times
        )

        flight_times = flight_times[
            (flight_times > 0) &
            (flight_times < 5)
        ]

        if len(flight_times) > 0:

            features[
                "flight_time_mean"
            ] = float(
                np.mean(flight_times)
            )

            features[
                "flight_time_std"
            ] = float(
                np.std(flight_times)
            )

            features[
                "flight_time_median"
            ] = float(
                np.median(flight_times)
            )

    # --------------------------------------------------------
    # TYPING EVENT RATE
    # --------------------------------------------------------

    duration = (
        keyboard["timestamp"].max()
        -
        keyboard["timestamp"].min()
    )

    if duration > 0:

        features[
            "typing_event_rate"
        ] = float(
            len(keyboard) / duration
        )

    return features


# ============================================================
# MOUSE FEATURES
# ============================================================

def extract_mouse_features(df):

    movement = df[
        df["event_type"] == "mouse_move"
    ].copy()

    clicks = df[
        df["event_type"] == "mouse_click"
    ].copy()

    features = {
        "mouse_velocity_mean": 0.0,
        "mouse_velocity_std": 0.0,
        "mouse_distance_mean": 0.0,
        "mouse_direction_change": 0.0,

        "click_interval_mean": 0.0,
        "click_interval_std": 0.0
    }

    # ========================================================
    # MOUSE MOVEMENT
    # ========================================================

    if len(movement) >= 2:

        movement = movement.sort_values(
            "timestamp"
        )

        x = pd.to_numeric(
            movement["mouse_x"],
            errors="coerce"
        ).values

        y = pd.to_numeric(
            movement["mouse_y"],
            errors="coerce"
        ).values

        t = movement[
            "timestamp"
        ].values

        # Remove invalid coordinates
        valid_coordinates = (
            np.isfinite(x) &
            np.isfinite(y) &
            np.isfinite(t)
        )

        x = x[valid_coordinates]
        y = y[valid_coordinates]
        t = t[valid_coordinates]

        if len(x) >= 2:

            dx = np.diff(x)
            dy = np.diff(y)
            dt = np.diff(t)

            valid = dt > 0

            dx = dx[valid]
            dy = dy[valid]
            dt = dt[valid]

            if len(dt) > 0:

                distances = np.sqrt(
                    dx ** 2 +
                    dy ** 2
                )

                velocities = (
                    distances / dt
                )

                valid_velocity = (
                    np.isfinite(velocities)
                )

                velocities = velocities[
                    valid_velocity
                ]

                distances = distances[
                    valid_velocity
                ]

                # ------------------------------------------------
                # VELOCITY
                # ------------------------------------------------

                if len(velocities) > 0:

                    features[
                        "mouse_velocity_mean"
                    ] = float(
                        np.mean(velocities)
                    )

                    features[
                        "mouse_velocity_std"
                    ] = float(
                        np.std(velocities)
                    )

                # ------------------------------------------------
                # DISTANCE
                # ------------------------------------------------

                if len(distances) > 0:

                    features[
                        "mouse_distance_mean"
                    ] = float(
                        np.mean(distances)
                    )

                # ------------------------------------------------
                # DIRECTION CHANGE
                # ------------------------------------------------

                if len(dx) >= 2:

                    angles = np.arctan2(
                        dy,
                        dx
                    )

                    unwrapped_angles = (
                        np.unwrap(angles)
                    )

                    angle_changes = np.diff(
                        unwrapped_angles
                    )

                    if len(
                        angle_changes
                    ) > 0:

                        features[
                            "mouse_direction_change"
                        ] = float(
                            np.mean(
                                np.abs(
                                    angle_changes
                                )
                            )
                        )

    # ========================================================
    # MOUSE CLICK FEATURES
    # ========================================================

    if len(clicks) >= 2:

        clicks = clicks.sort_values(
            "timestamp"
        )

        click_times = clicks[
            clicks["key_state"] == "down"
        ]["timestamp"].values

        if len(click_times) >= 2:

            intervals = np.diff(
                click_times
            )

            intervals = intervals[
                (intervals > 0) &
                (intervals < 10)
            ]

            if len(intervals) > 0:

                features[
                    "click_interval_mean"
                ] = float(
                    np.mean(intervals)
                )

                features[
                    "click_interval_std"
                ] = float(
                    np.std(intervals)
                )

    return features


# ============================================================
# EXTRACT FEATURES FROM ONE WINDOW
# ============================================================

def extract_window_features(window):

    keyboard_features = (
        extract_keyboard_features(
            window
        )
    )

    mouse_features = (
        extract_mouse_features(
            window
        )
    )

    return {
        **keyboard_features,
        **mouse_features
    }


# ============================================================
# SPLIT SESSION INTO WINDOWS
# ============================================================

def create_windows(
    csv_path,
    window_seconds=WINDOW_SECONDS
):

    df = load_data(csv_path)

    start_time = df[
        "timestamp"
    ].min()

    end_time = df[
        "timestamp"
    ].max()

    total_duration = (
        end_time -
        start_time
    )

    windows = []

    current_start = start_time

    while current_start < end_time:

        current_end = (
            current_start +
            window_seconds
        )

        window = df[
            (df["timestamp"] >= current_start)
            &
            (df["timestamp"] < current_end)
        ].copy()

        # Ignore extremely small windows
        if len(window) >= 5:

            features = (
                extract_window_features(
                    window
                )
            )

            features[
                "window_start"
            ] = current_start - start_time

            features[
                "window_end"
            ] = current_end - start_time

            windows.append(
                features
            )

        current_start = current_end

    return pd.DataFrame(windows)


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(csv_path):

    print()
    print("=" * 60)
    print("PROCESSING")
    print("=" * 60)

    print(
        f"File: {csv_path}"
    )

    windows = create_windows(
        csv_path
    )

    print(
        f"Windows created: {len(windows)}"
    )

    return windows


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def process_and_save(
    csv_path,
    output_path
):

    windows = process_file(
        csv_path
    )

    if windows.empty:

        print(
            "WARNING: No usable windows found."
        )

        return

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    windows.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved to: {output_path}"
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    input_file = (
        "data/raw/owner/"
        "day01_session01.csv"
    )

    output_file = (
        "data/processed/"
        "day01_session01_features.csv"
    )

    process_and_save(
        input_file,
        output_file
    )

    print()
    print(
        "Feature extraction completed."
    )