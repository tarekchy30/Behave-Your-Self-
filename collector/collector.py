import csv
import os
import time
from pynput import keyboard, mouse


# ============================================================
# CONFIGURATION
# ============================================================

USER_TYPE = "owner"

OUTPUT_DIR = os.path.join(
    "data",
    "raw",
    USER_TYPE
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# AUTOMATIC SESSION NUMBER
# ============================================================

def get_next_session():

    # Look for existing day/session files
    existing_files = os.listdir(OUTPUT_DIR)

    # Example:
    # day01_session01.csv
    # day01_session02.csv
    # day02_session01.csv

    used_sessions = []

    for filename in existing_files:

        if not filename.endswith(".csv"):
            continue

        if not filename.startswith("day"):
            continue

        try:

            name = filename.replace(
                ".csv",
                ""
            )

            parts = name.split("_")

            day_number = int(
                parts[0].replace(
                    "day",
                    ""
                )
            )

            session_number = int(
                parts[1].replace(
                    "session",
                    ""
                )
            )

            used_sessions.append(
                (
                    day_number,
                    session_number
                )
            )

        except Exception:
            continue

    # --------------------------------------------------------
    # First session
    # --------------------------------------------------------

    if not used_sessions:

        return 1, 1

    # --------------------------------------------------------
    # Continue from last session
    # --------------------------------------------------------

    last_day, last_session = max(
        used_sessions
    )

    # Maximum 2 sessions per day
    if last_session < 2:

        return (
            last_day,
            last_session + 1
        )

    # Start next day
    return (
        last_day + 1,
        1
    )


DAY_NUMBER, SESSION_NUMBER = get_next_session()

SESSION_NAME = (
    f"day{DAY_NUMBER:02d}_"
    f"session{SESSION_NUMBER:02d}"
)


# ============================================================
# OUTPUT FILE
# ============================================================

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"{SESSION_NAME}.csv"
)


# ============================================================
# STATE
# ============================================================

running = True


# ============================================================
# CSV FILE
# ============================================================

csv_file = open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_file)

writer.writerow([
    "timestamp",
    "event_type",
    "key_state",
    "mouse_x",
    "mouse_y",
    "mouse_button"
])


# ============================================================
# TIMESTAMP
# ============================================================

def timestamp():

    return time.time()


# ============================================================
# KEYBOARD
# ============================================================

def on_key_press(key):

    writer.writerow([
        timestamp(),
        "keyboard",
        "down",
        "",
        "",
        ""
    ])

    csv_file.flush()


def on_key_release(key):

    writer.writerow([
        timestamp(),
        "keyboard",
        "up",
        "",
        "",
        ""
    ])

    csv_file.flush()


# ============================================================
# MOUSE MOVEMENT
# ============================================================

def on_move(x, y):

    writer.writerow([
        timestamp(),
        "mouse_move",
        "",
        x,
        y,
        ""
    ])

    csv_file.flush()


# ============================================================
# MOUSE CLICK
# ============================================================

def on_click(
    x,
    y,
    button,
    pressed
):

    writer.writerow([
        timestamp(),
        "mouse_click",
        "down" if pressed else "up",
        x,
        y,
        str(button)
    ])

    csv_file.flush()


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 60)
print("BEHAVIORAL DATA COLLECTOR")
print("=" * 60)

print(
    f"User type : {USER_TYPE}"
)

print(
    f"Session   : {SESSION_NAME}"
)

print(
    f"Output    : {OUTPUT_FILE}"
)

print()

print(
    "The collector records behavioral metadata."
)

print(
    "It does NOT record the characters you type."
)

print()

print(
    "Use the computer normally."
)

print(
    "Type normally, browse normally, move the mouse normally."
)

print()

print(
    "Press Ctrl+C to stop."
)

print()

# ============================================================
# START LISTENERS
# ============================================================

keyboard_listener = keyboard.Listener(
    on_press=on_key_press,
    on_release=on_key_release
)

mouse_listener = mouse.Listener(
    on_move=on_move,
    on_click=on_click
)

keyboard_listener.start()
mouse_listener.start()


# ============================================================
# KEEP RUNNING
# ============================================================

try:

    while running:

        time.sleep(1)

except KeyboardInterrupt:

    print()
    print(
        "Stopping collector..."
    )


# ============================================================
# CLEANUP
# ============================================================

finally:

    keyboard_listener.stop()
    mouse_listener.stop()

    csv_file.close()

    print()
    print(
        "Saved behavioral data:"
    )

    print(
        OUTPUT_FILE
    )

    print()