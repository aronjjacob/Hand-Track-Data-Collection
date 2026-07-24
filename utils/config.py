"""
config.py
=========
Central place for constants used across the FSL data collection app:
paths, recording parameters, and the keyboard -> sign label mapping.

Adding a new FSL sign in the future only requires adding one line to
FSL_MEDICAL_SIGNS below. No other file needs to change.
"""

import os


# ==========================
# Dataset paths
# ==========================

# Root folder where all sign datasets are stored.
# Resolves to HAND-TRACK/dataset regardless of where the script is run from.
DATASET_ROOT = os.path.join(
    os.path.dirname(            # utils/  -> HAND-TRACK/
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "dataset"
)


# ==========================
# Sequence recording settings
# ==========================

# Number of consecutive frames that make up one gesture sequence.
# Each saved file has shape (SEQUENCE_LENGTH, 21, 3).
SEQUENCE_LENGTH = 30


# ==========================
# Filipino Sign Language Medical Signs
# ==========================
# Maps a cv2.waitKey() key code (int) -> FSL medical sign label (str).
# Keys are lowercase for letters, since cv2.waitKey(1) & 0xFF returns
# lowercase ASCII codes for unshifted letter key presses.
#
# To add a new sign later: pick an unused key and add one line here.
FSL_MEDICAL_SIGNS = {
    ord("1"): "TULONG",
    ord("2"): "EMERHENSIYA",
    ord("3"): "OSPITAL",
    ord("4"): "DOKTOR",
    ord("5"): "AMBULANSIYA",
    ord("6"): "SAKIT",
    ord("7"): "LAGNAT",
    ord("8"): "HIRAP_HUMINGA",
    ord("9"): "SAKIT_NG_ULO",
    ord("a"): "SAKIT_NG_TIYAN",
    ord("b"): "GAMOT",
    ord("c"): "ALERHIYA",
    ord("d"): "MAY_SAKIT",
    ord("e"): "OO",
    ord("f"): "HINDI",
    ord("g"): "SALAMAT",
}