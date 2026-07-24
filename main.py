import cv2
import mediapipe as mp
import os
import time
import numpy as np

from utils.landmarks import extract_landmarks, print_landmarks
from utils.data_collector import DataCollector
from utils.sign_manager import SignManager
from utils.config import SEQUENCE_LENGTH


# ==========================
# MediaPipe Setup (UNCHANGED)
# ==========================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode


# Hand skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17)                                   # Palm
]


# ==========================
# Load Hand Model (UNCHANGED)
# ==========================

model_path = os.path.join(
    os.path.dirname(__file__),
    "hand_landmarker.task"
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

detector = HandLandmarker.create_from_options(options)


# ==========================
# MediaPipe Detection (UNCHANGED)
# ==========================

def mediapipe_detection(frame, timestamp):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    results = detector.detect_for_video(
        mp_image,
        timestamp
    )

    return results


# ==========================
# Draw Hand (UNCHANGED)
# ==========================

def draw_hand(frame, landmarks, label):

    h, w, _ = frame.shape

    points = []

    for lm in landmarks:

        x = int(lm.x * w)
        y = int(lm.y * h)

        points.append((x, y))

    # White skeleton lines
    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 255),
            3
        )

    # Green joints
    for x, y in points:

        cv2.circle(
            frame,
            (x, y),
            8,
            (0, 255, 0),
            -1
        )

    # Hand label near wrist
    wrist_x, wrist_y = points[0]

    cv2.putText(
        frame,
        label,
        (wrist_x - 40, wrist_y - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


# ==========================
# Draw UI Overlay
# ==========================

def draw_ui(frame, collector, sign_manager, is_recording, frame_count, save_flash_timer):
    """
    Draws the data collection status panel onto the OpenCV frame.

    Idle:
        Current Sign : DOKTOR
        Sequences Saved : 25
        Status : Idle

    Recording:
        Current Sign : DOKTOR
        Status : Recording...
        Frame : 23 / 45

    Just after a save (flashed for ~1 second):
        Saved: sequence_026.npy

    Parameters:
        frame            -> current BGR video frame to draw on
        collector        -> DataCollector instance (label + counts)
        sign_manager     -> SignManager instance (not strictly needed here
                             since collector.sign_label mirrors it, but
                             kept for clarity / future use)
        is_recording     -> bool, whether a sequence is currently recording
        frame_count      -> int, frames captured so far in this recording
        save_flash_timer -> float timestamp of last save (time.time())
                             0.0 if no save has happened yet
    """

    # Semi-transparent dark background panel at top-left
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (460, 150),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Current sign label
    cv2.putText(
        frame,
        f"Current Sign : {collector.sign_label}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 200, 255),     # Orange-yellow
        2
    )

    # Sequences saved for this sign
    cv2.putText(
        frame,
        f"Sequences Saved : {collector.sequences_saved}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (200, 200, 200),   # Light grey
        2
    )

    # Status line: Idle vs Recording (with live frame count)
    if is_recording:
        status_text = "Status : Recording..."
        frame_text = f"Frame : {frame_count} / {SEQUENCE_LENGTH}"
    else:
        status_text = "Status : Idle"
        frame_text = ""

    status_color = (0, 0, 255) if is_recording else (0, 255, 0)

    cv2.putText(
        frame,
        status_text,
        (20, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        2
    )

    if frame_text:
        cv2.putText(
            frame,
            frame_text,
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )

    # Flash "Saved: sequence_NNN.npy" for ~1 second after each save.
    # Drawn below the panel so it doesn't overlap the status lines.
    flash_duration = 1.0

    if save_flash_timer > 0 and (time.time() - save_flash_timer) < flash_duration:

        cv2.rectangle(frame, (10, 160), (460, 195), (0, 0, 0), -1)

        cv2.putText(
            frame,
            collector.last_status,
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 100),     # Bright green
            2
        )


# ==========================
# Main
# ==========================

def main():

    # ---- Sign selection is now live, driven by SignManager ----
    # No more manually editing SIGN_LABEL and restarting the app.
    sign_manager = SignManager()

    # DataCollector starts on whatever sign SignManager defaults to
    collector = DataCollector(sign_label=sign_manager.current_label)

    print("[Main] FSL Medical Gesture Recording Session Started")
    print(f"       Starting sign : {collector.sign_label}")
    print(f"       Existing sequences : {collector.sequences_saved}")
    print("       Sign keys:")
    for key_char, label in sign_manager.key_label_pairs():
        print(f"         [{key_char.upper()}] -> {label}")
    print("       [R] Record a 45-frame sequence")
    print("       [Q] Quit")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    timestamp = 0
    previous_time = 0

    # Tracks when the last save happened, for the flash overlay
    save_flash_timer = 0.0

    # ==========================
    # Sequence recording state
    # ==========================
    # is_recording   -> True while capturing the 45 frames of one gesture
    # sequence_buffer -> list of (21, 3) arrays collected so far
    is_recording = False
    sequence_buffer = []

    # Holds the most recently extracted landmark array across the loop.
    # Updated every frame when a hand is detected; None when no hand visible.
    current_landmarks = None
    current_label = None

    while True:

        success, frame = cap.read()

        if not success:
            print("Camera error")
            break

        # Mirror the frame so left/right feels natural
        frame = cv2.flip(frame, 1)

        # Run MediaPipe detection on the current frame (unchanged)
        results = mediapipe_detection(frame, timestamp)

        timestamp += 1

        # Reset per-frame landmark cache
        current_landmarks = None
        current_label = None

        if results.hand_landmarks:

            for index, hand_landmarks in enumerate(results.hand_landmarks):

                # Resolve MediaPipe handedness and correct for mirror flip
                label = results.handedness[index][0].category_name

                if label == "Left":
                    label = "Right"
                else:
                    label = "Left"

                # Draw hand skeleton — unchanged from original
                draw_hand(frame, hand_landmarks, label)

                # Extract (21, 3) NumPy array — unchanged extraction logic
                landmarks_array = extract_landmarks(hand_landmarks)

                # Cache the most recent hand's data.
                # If two hands are detected, the last one in the loop wins
                # (FSL medical signs here are treated as one-handed).
                current_landmarks = landmarks_array
                current_label = label

        # ==========================
        # Sequence recording tick
        # ==========================
        # If currently recording, capture one frame per loop iteration
        # regardless of hand visibility, so the sequence always has
        # exactly SEQUENCE_LENGTH frames and stays time-aligned.
        # A frame with no hand detected is stored as all-zeros so the
        # array shape/timing stays consistent for CNN-LSTM training.
        if is_recording:

            if current_landmarks is not None:
                sequence_buffer.append(current_landmarks)
            else:
                sequence_buffer.append(
                    np.zeros((21, 3), dtype=np.float32)
                )
                print("[Main] Warning: no hand detected on a recorded frame.")

            # Once we have all frames, save the full sequence
            if len(sequence_buffer) == SEQUENCE_LENGTH:

                sequence_array = np.stack(sequence_buffer, axis=0)  # (45, 21, 3)

                collector.save_sequence(sequence_array)

                save_flash_timer = time.time()
                is_recording = False
                sequence_buffer = []

        # Draw the UI status panel on top of the video frame
        draw_ui(
            frame,
            collector,
            sign_manager,
            is_recording,
            len(sequence_buffer),
            save_flash_timer
        )

        # FPS display (top-right, unchanged from original)
        current_time = time.time()
        fps = 1 / (current_time - previous_time) if current_time != previous_time else 0
        previous_time = current_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (1100, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.imshow("FSL Hand Tracker", frame)

        # ==========================
        # Keyboard Controls
        # ==========================

        key = cv2.waitKey(1) & 0xFF

        # Sign selection keys (1-9, A-G) — only allowed while idle so a
        # sign switch can't corrupt a sequence that's mid-recording.
        if sign_manager.is_sign_key(key):

            if is_recording:
                print("[Main] Ignored sign switch — recording in progress.")
            else:
                new_label = sign_manager.select(key)
                collector.set_label(new_label)

        # R — Start recording one 45-frame gesture sequence
        elif key == ord("r"):

            if is_recording:
                print("[Main] Already recording — please wait.")
            else:
                is_recording = True
                sequence_buffer = []
                print(f"[Main] Recording started for '{collector.sign_label}'...")

        # Q — Quit
        elif key == ord("q"):

            print(
                f"[Main] Session ended. "
                f"Sequences saved for '{collector.sign_label}': "
                f"{collector.sequences_saved}"
            )
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()