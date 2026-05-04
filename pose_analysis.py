import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load model once
def create_pose_landmarker():
    base_options = python.BaseOptions(
        model_asset_path="pose_landmarker_lite.task"
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    return vision.PoseLandmarker.create_from_options(options)


def extract_pose_sequence(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    landmarker = create_pose_landmarker()

    frame_idx = 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MP Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp REQUIRED for VIDEO mode
        timestamp_ms = int(frame_idx * 33)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            sequence.append({
                "right_wrist_x": landmarks[16].x,
                "right_wrist_y": landmarks[16].y,
                "right_elbow_x": landmarks[14].x,
                "right_elbow_y": landmarks[14].y,
                "right_shoulder_x": landmarks[12].x,
                "right_shoulder_y": landmarks[12].y
            })

        frame_idx += 1

    cap.release()
    landmarker.close()

    return sequence


def compute_features(sequence):
    velocities = []

    for i in range(1, len(sequence)):
        prev = sequence[i - 1]
        curr = sequence[i]

        velocity = curr["right_wrist_x"] - prev["right_wrist_x"]
        velocities.append(velocity)

    avg_velocity = np.mean(velocities) if velocities else 0

    return {
        "avg_velocity": avg_velocity
    }


def classify_strike(features):
    v = features["avg_velocity"]

    if v > 0.02:
        return "Winkel 1 (Diagonal)"
    elif v < -0.02:
        return "Winkel 2 (Gegendiagonal)"
    else:
        return "Unklar / anderer Winkel"