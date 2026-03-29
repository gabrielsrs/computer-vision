import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

CSV_FILE = "trained_data/hand_landmarks_dataset.csv"
MODEL_FILE = "models/own_models/gesture_model.pkl"
LABEL_ENCODER_FILE = "models/own_models/label_encoder.pkl"

print("=" * 50)
print("HAND GESTURE RECOGNITION MODEL TRAINING")
print("=" * 50)

df = pd.read_csv(CSV_FILE)
print(f"\nDataset loaded: {len(df)} samples")
print(f"Features: {df.shape[1] - 1}")
print(f"Classes: {df['label'].unique()}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")

X = df.drop("label", axis=1)
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nLabel encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

print("\nTraining Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": clf.feature_importances_}
).sort_values("importance", ascending=False)
print("\nTop 10 Important Features:")
print(feature_importance.head(10).to_string(index=False))

os.makedirs("models", exist_ok=True)
joblib.dump(clf, MODEL_FILE)
joblib.dump(le, LABEL_ENCODER_FILE)
print(f"\nModel saved to: {MODEL_FILE}")
print(f"Label encoder saved to: {LABEL_ENCODER_FILE}")
