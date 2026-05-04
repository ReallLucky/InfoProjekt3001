import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =========================
# LOAD DATASET
# =========================
DATA_PATH = "strike1_dataset.csv"

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# =========================
# SPLIT FEATURES / LABEL
# =========================
X = df.drop(columns=["label"])
y = df["label"]

# =========================
# ENCODE LABELS
# =========================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# =========================
# NORMALIZE FEATURES (IMPORTANT for k-NN)
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# TRAIN k-NN MODEL
# =========================
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance"
)

knn.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n====================")
print(f"Accuracy: {accuracy:.4f}")
print("====================\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

# =========================
# SAVE MODEL
# =========================
joblib.dump(knn, "eskrima_knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\n✅ Model saved successfully!")