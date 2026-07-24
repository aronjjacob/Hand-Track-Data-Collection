import os
import numpy as np

from utils.config import DATASET_ROOT, SEQUENCE_LENGTH


# ==========================
# DataCollector Class
# ==========================

class DataCollector:
    """
    Manages saving FSL gesture *sequences* to disk.

    A gesture sequence is SEQUENCE_LENGTH consecutive frames of hand
    landmarks. Each frame is a (21, 3) array, so one saved gesture has
    shape:

        (SEQUENCE_LENGTH, 21, 3)   e.g. (45, 21, 3)

    Responsibilities:
        - Determine the correct save path under dataset/<sign_label>/
        - Auto-create folders if they do not exist
        - Auto-number files as sequence_001.npy, sequence_002.npy, ...
        - Track how many sequences exist for the *current* sign
        - Support switching signs mid-session via set_label(), re-scanning
          the new folder so numbering is always correct
        - Report save status back to the caller for the OpenCV UI overlay

    One instance of DataCollector is shared for the whole session;
    set_label() is called every time the user switches signs so the
    rest of the app never has to create a new instance.
    """

    def __init__(self, sign_label: str):
        """
        Parameters:
            sign_label -> Initial FSL sign to record, e.g. "TULONG"
                          Used as the subfolder name under dataset/
        """

        # Store and normalize the label to uppercase for consistency
        self.sign_label = sign_label.upper()

        # Full path to this sign's folder, e.g. dataset/TULONG/
        self.sign_folder = os.path.join(DATASET_ROOT, self.sign_label)

        # Create dataset/ and dataset/<sign_label>/ if they don't exist
        os.makedirs(self.sign_folder, exist_ok=True)

        # Count of sequences already saved for the current sign.
        # Scanned from disk so numbering is correct even after restarts
        # or after switching back to a sign that already has data.
        self.sequences_saved = self._count_existing_sequences()

        # Human-readable status shown briefly on screen after each save
        self.last_status = ""

    # ==========================
    # Count Existing Sequences
    # ==========================

    def _count_existing_sequences(self) -> int:
        """
        Counts how many sequence_*.npy files already exist in the
        current sign's folder, so numbering continues correctly
        (e.g. next save after sequence_005.npy is sequence_006.npy).
        """

        existing = [
            f for f in os.listdir(self.sign_folder)
            if f.startswith("sequence_") and f.endswith(".npy")
        ]

        return len(existing)

    # ==========================
    # Save Sequence
    # ==========================

    def save_sequence(self, sequence_array: np.ndarray) -> str:
        """
        Saves one complete gesture sequence.

        Data flow:
            45 frames of extract_landmarks() output
                ↓ stacked by main.py into one array
            NumPy array (SEQUENCE_LENGTH, 21, 3) float32
                ↓ save_sequence() here
            dataset/<SIGN_LABEL>/sequence_NNN.npy on disk

        Parameters:
            sequence_array -> np.ndarray, shape (SEQUENCE_LENGTH, 21, 3)

        Returns:
            str -> filename that was saved, e.g. "sequence_026.npy"

        Raises:
            ValueError if sequence_array is not the expected shape.
        """

        expected_shape = (SEQUENCE_LENGTH, 21, 3)

        if sequence_array.shape != expected_shape:
            raise ValueError(
                f"Expected sequence_array shape {expected_shape}, "
                f"got {sequence_array.shape}"
            )

        # Increment count first so numbering starts at 001
        self.sequences_saved += 1

        sample_number = str(self.sequences_saved).zfill(3)
        filename = f"sequence_{sample_number}.npy"
        filepath = os.path.join(self.sign_folder, filename)

        # np.save() serializes the array to binary .npy format, preserving
        # exact float32 precision and shape for later CNN-LSTM training.
        np.save(filepath, sequence_array)

        self.last_status = f"Saved: {filename}"

        print(f"[DataCollector] Saved -> {filepath}")

        return filename

    # ==========================
    # Change Label Mid-Session
    # ==========================

    def set_label(self, new_label: str):
        """
        Switches the active sign label without restarting the program.

        Creates the new folder if needed and re-scans it so the
        sequence counter reflects that sign's existing data.

        Parameters:
            new_label -> new FSL sign label, e.g. "DOKTOR"
        """

        self.sign_label = new_label.upper()
        self.sign_folder = os.path.join(DATASET_ROOT, self.sign_label)

        os.makedirs(self.sign_folder, exist_ok=True)

        # Re-count from the new folder so numbering is correct
        self.sequences_saved = self._count_existing_sequences()

        self.last_status = f"Label: {self.sign_label}"

        print(
            f"[DataCollector] Label changed -> {self.sign_label} "
            f"({self.sequences_saved} existing sequences)"
        )