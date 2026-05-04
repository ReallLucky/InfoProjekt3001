import cv2
import numpy as np
import pandas as pd
import os
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# =========================
# CONFIG
# =========================
MODEL_PATH = "pose_landmarker_full.task"
WINDOW_SIZE = 15
STEP_SIZE = 5

# ✅ YOUR FOLDERS
DATA_FOLDERS = {
    "Clip Idle": "idle",

    "Clips Richtig 1": "strike_1_correct",
    "Clips Richtig 2": "strike_2_correct",

    "Clip Schulter hinten": "strike_1_shoulder_back",

    "Clip zu früh laufen 2": "strike_2_too_early",
    "Clips zu früh laufen 1": "strike_1_too_early"
}

# =========================
# MEDIAPIPE (IMAGE MODE)
# =========================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)

landmarker = vision.PoseLandmarker.create_from_options(options)

# =========================
# EXTRACT FRAMES
# =========================
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            row = []
            for p in lm:
                row.extend([p.x, p.y, p.z])
            frames.append(row)

    cap.release()
    return frames

# =========================
# BUILD SEQUENCES
# =========================
def build_sequences(frames, label):
    sequences = []

    for i in range(0, len(frames) - WINDOW_SIZE, STEP_SIZE):
        window = frames[i:i + WINDOW_SIZE]
        sequence = np.array(window).flatten()
        sequence = np.append(sequence, label)
        sequences.append(sequence)

    return sequences

# =========================
# DATASET
# =========================
def build_dataset():
    all_data = []

    for folder, label in DATA_FOLDERS.items():

        if not os.path.exists(folder):
            print("Missing:", folder)
            continue

        print("\nProcessing", folder)

        for file in os.listdir(folder):
            if file.lower().endswith((".mp4", ".mov", ".avi")):

                path = os.path.join(folder, file)
                print(" →", file)

                frames = extract_frames(path)

                if len(frames) >= WINDOW_SIZE:
                    seqs = build_sequences(frames, label)
                    all_data.extend(seqs)

    return all_data

# =========================
# TRAIN
# =========================
def train_model(data):

    feature_count = 33 * 3 * WINDOW_SIZE

    columns = [f"f{i}" for i in range(feature_count)]
    columns.append("label")

    df = pd.DataFrame(data, columns=columns)

    X = df.drop(columns=["label"])
    y = df["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nClasses:", label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    joblib.dump(knn, "eskrima_knn_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")

    print("\n✅ Model saved!")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    dataset = build_dataset()

    if len(dataset) == 0:
        print("No data found.")
    else:
        train_model(dataset)