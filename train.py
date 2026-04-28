import cv2
import numpy as np
import os
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical


# =========================
# CONFIG
# =========================
MODEL_PATH = "pose_landmarker_full.task"

WINDOW_SIZE = 12        # optimized for fast strikes
STEP_SIZE   = 2         # dense sampling

ANALYSIS_FPS = 15
RESIZE_W, RESIZE_H = 640, 360

N_LANDMARKS = 33
N_COORDS = 3

RAW_FEATURES = N_LANDMARKS * N_COORDS
TOTAL_FEATURES = RAW_FEATURES * 2


DATA_FOLDERS = {
    "Clip Idle":              "idle",
    "Clips Richtig 1":        "strike_1_correct",
    "Clips Richtig 2":        "strike_2_correct",
    "Clip Schulter hinten":   "strike_1_shoulder_back",
    "Clip zu früh laufen 2":  "strike_2_too_early",
    "Clips zu früh laufen 1": "strike_1_too_early",
}


# =========================
# MEDIAPIPE
# =========================
def make_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


# =========================
# NORMALIZATION
# =========================
def normalize_landmarks(row):
    pts = row.reshape(N_LANDMARKS, N_COORDS)

    left_hip = pts[23]
    right_hip = pts[24]
    left_shoulder = pts[11]

    origin = (left_hip + right_hip) / 2.0
    scale = np.linalg.norm(left_shoulder - origin) + 1e-6

    return ((pts - origin) / scale).flatten()


# =========================
# VIDEO → LANDMARKS
# =========================
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(source_fps / ANALYSIS_FPS)))

    landmarker = make_landmarker()

    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            small = cv2.resize(frame, (RESIZE_W, RESIZE_H))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((frame_idx / source_fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                raw = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
                frames.append(normalize_landmarks(raw))

        frame_idx += 1

    cap.release()
    landmarker.close()

    return frames


# =========================
# VELOCITY
# =========================
def add_velocity(frames):
    if len(frames) == 0:
        return []

    out = []
    for i in range(len(frames)):
        vel = frames[i] - frames[i - 1] if i > 0 else np.zeros(RAW_FEATURES)
        out.append(np.concatenate([frames[i], vel]))

    return out


# =========================
# AUGMENTATION
# =========================
def augment(seq):
    out = [seq]

    noisy = seq.copy()
    noisy[:, :RAW_FEATURES] += np.random.normal(0, 0.01, noisy[:, :RAW_FEATURES].shape)
    out.append(noisy)

    flipped = seq.copy()
    flipped[:, 0:RAW_FEATURES:3] *= -1
    flipped[:, RAW_FEATURES::3] *= -1
    out.append(flipped)

    return out


# =========================
# SEQUENCES
# =========================
def build_sequences(frames, label):
    frames = add_velocity(frames)

    if len(frames) == 0:
        return []

    # PAD instead of skipping
    if len(frames) < WINDOW_SIZE:
        pad = np.tile(frames[-1], (WINDOW_SIZE - len(frames), 1))
        frames = np.vstack([frames, pad])

    sequences = []

    for i in range(0, len(frames) - WINDOW_SIZE + 1, STEP_SIZE):
        window = np.array(frames[i:i + WINDOW_SIZE])

        for aug in augment(window):
            sequences.append((aug, label))

    return sequences


# =========================
# DATASET
# =========================
def build_dataset():
    X, y = [], []

    for folder, label in DATA_FOLDERS.items():
        if not os.path.exists(folder):
            print("Missing:", folder)
            continue

        print("\nFolder:", folder)

        for file in os.listdir(folder):
            if not file.endswith((".mp4", ".avi", ".mov")):
                continue

            path = os.path.join(folder, file)

            print("Processing:", file)

            frames = extract_frames(path)

            seqs = build_sequences(frames, label)

            for s, l in seqs:
                X.append(s)
                y.append(l)

    return np.array(X), np.array(y)


# =========================
# MODEL
# =========================
def build_model(n_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(WINDOW_SIZE, TOTAL_FEATURES)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.2),

        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================
# TRAIN
# =========================
def train(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_classes = len(le.classes_)

    print("\nClasses:", le.classes_)
    print("Dataset:", X.shape)

    N, T, F = X.shape
    X = X.reshape(N, T * F)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(N, T, F)

    y_cat = to_categorical(y_enc)

    X_train, X_test, y_train, y_test, yti, yte = train_test_split(
        X, y_cat, y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc
    )

    model = build_model(n_classes)

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5),
        ModelCheckpoint("best_model.keras", save_best_only=True)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=60,
        batch_size=32,
        callbacks=callbacks
    )

    preds = np.argmax(model.predict(X_test), axis=1)

    print("\nAccuracy:", accuracy_score(yte, preds))
    print(classification_report(yte, preds, target_names=le.classes_))

    model.save("eskrima_model.keras")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le, "label_encoder.pkl")

    print("\nSaved!")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("Training Eskrima model...")

    X, y = build_dataset()

    if len(X) == 0:
        print("No data found.")
    else:
        train(X, y)