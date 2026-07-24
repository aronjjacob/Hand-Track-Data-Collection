"""
sign_manager.py
================
Encapsulates the keyboard-key -> FSL sign label mapping so main.py
never has to hardcode "if key == ord('1'): label = 'TULONG'" logic.

SignManager only tracks *which sign is currently selected*. It does
NOT touch the filesystem or DataCollector directly — main.py is
responsible for calling collector.set_label(...) when the selection
changes. This keeps responsibilities cleanly separated:
    SignManager   -> "what sign is selected, and which keys select it"
    DataCollector -> "where does that sign's data get saved"
"""

from utils.config import FSL_MEDICAL_SIGNS


class SignManager:
    """
    Tracks the currently selected FSL sign and resolves key presses
    to sign labels using the FSL_MEDICAL_SIGNS mapping from config.py.
    """

    def __init__(self, sign_map: dict = None):
        # Allow a custom map to be injected (useful for testing);
        # defaults to the predefined medical sign list.
        self.sign_map = sign_map or FSL_MEDICAL_SIGNS

        if not self.sign_map:
            raise ValueError("SignManager requires a non-empty sign map.")

        # Default to the first sign in the map on startup so there is
        # always a valid current_label before the user presses a key.
        first_key = next(iter(self.sign_map))
        self.current_key = first_key
        self.current_label = self.sign_map[first_key]

    # ==========================
    # Lookups
    # ==========================

    def is_sign_key(self, key: int) -> bool:
        """True if `key` (from cv2.waitKey() & 0xFF) selects a sign."""
        return key in self.sign_map

    def select(self, key: int):
        """
        Attempts to select a new sign based on a key press.

        Returns:
            str  -> the new current_label, if `key` was a valid sign key
            None -> if `key` did not match any sign (caller should ignore)
        """
        if key not in self.sign_map:
            return None

        self.current_key = key
        self.current_label = self.sign_map[key]
        return self.current_label

    def all_labels(self):
        """Returns every sign label this manager knows about."""
        return list(self.sign_map.values())

    def key_label_pairs(self):
        """
        Returns [(key_as_char, label), ...] sorted for display purposes,
        e.g. for printing a legend at startup.
        """
        pairs = [(chr(k), label) for k, label in self.sign_map.items()]
        return sorted(pairs, key=lambda pair: pair[0])