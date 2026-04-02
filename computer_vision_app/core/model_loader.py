import os
import joblib
import mediapipe as mp

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MP_MODEL_PATH = os.path.join("models", "gesture_recognizer.task")
# CUSTOM_MODEL_PATH = os.path.join("models", "gesture_model.joblib")
# ENCODER_PATH = os.path.join("models", "label_encoder.joblib")

MP_MODEL_PATH = "models/gesture_recognizer.task"
CUSTOM_MODEL_PATH = "models/gesture_model.joblib"
ENCODER_PATH = "models/label_encoder.joblib"


def check_models_exist():
    return all(
        os.path.exists(p) for p in [MP_MODEL_PATH, CUSTOM_MODEL_PATH, ENCODER_PATH]
    )


def load_models():
    if not check_models_exist():
        raise FileNotFoundError("One or more model files are missing (.task, .joblib)")

    clf = joblib.load(CUSTOM_MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    recognizer = GestureRecognizer.create_from_options(options)

    return {
        "clf": clf,
        "label_encoder": label_encoder,
        "recognizer": recognizer,
    }
