import cv2
import numpy as np
import mediapipe as mp

# Correct for mediapipe 0.10.33
mp_pose = mp.solutions.pose


def extract_pose_sequence(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    # Use updated Pose constructor (explicit args = more stable)
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        count = 0

        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                sequence.append({
                    "right_wrist_x": landmarks[16].x,
                    "right_wrist_y": landmarks[16].y,
                    "right_elbow_x": landmarks[14].x,
                    "right_elbow_y": landmarks[14].y,
                    "right_shoulder_x": landmarks[12].x,
                    "right_shoulder_y": landmarks[12].y
                })

            count += 1

    cap.release()
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