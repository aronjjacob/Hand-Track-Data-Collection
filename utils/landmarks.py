import numpy as np


# ==========================
# Landmark Extraction
# ==========================

def extract_landmarks(hand_landmarks):
    """
    Converts 21 MediaPipe NormalizedLandmark objects into a
    NumPy array of shape (21, 3).

    Each row represents one landmark:
        [x, y, z]

    x, y  -> normalized 0.0 to 1.0, relative to frame size
    z     -> depth relative to the wrist
              negative = closer to camera
              positive = farther from camera

    Parameters:
        hand_landmarks  -> list of 21 NormalizedLandmark objects
                           returned by MediaPipe HandLandmarker

    Returns:
        np.ndarray of shape (21, 3), dtype float32
    """

    landmarks_list = []

    for lm in hand_landmarks:

        # Each lm has .x .y .z as Python floats
        landmarks_list.append(
            [lm.x, lm.y, lm.z]
        )

    # Stack into (21, 3) float32 array
    # float32 saves disk space vs float64 with no meaningful precision loss
    # for landmark coordinates which only need ~3 decimal places
    landmarks_array = np.array(
        landmarks_list,
        dtype=np.float32
    )

    return landmarks_array


# ==========================
# Print Landmarks (terminal debug)
# ==========================

def print_landmarks(label, landmarks_array):
    """
    Prints all 21 landmarks for a detected hand to the terminal.
    Used for visual debugging during development.

    Parameters:
        label           -> "Left" or "Right"
        landmarks_array -> np.ndarray of shape (21, 3)
    """

    print(f"\nHand: {label}")

    for index, (x, y, z) in enumerate(landmarks_array):

        print(
            f"  Landmark {index:>2}: "
            f"x = {x:.3f}  "
            f"y = {y:.3f}  "
            f"z = {z:.3f}"
        )