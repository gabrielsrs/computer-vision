import cv2
import mediapipe as mp
import joblib
import numpy as np

model = joblib.load("models/own_models/gesture_model.pkl")
label_encoder = joblib.load("models/own_models/label_encoder.pkl")

base_options = mp.tasks.BaseOptions(model_asset_path="models/gesture_recognizer.task")
VisionRunningMode = mp.tasks.vision.RunningMode
options = mp.tasks.vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=VisionRunningMode.IMAGE,
)
recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]


def draw_hand_landmarks(frame, hand_landmarks, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        cv2.line(frame, start_point, end_point, color, 2)
    for idx, landmark in enumerate(hand_landmarks):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)


def extract_landmarks(hand_landmarks):
    coords = []
    for landmark in hand_landmarks:
        coords.extend([landmark.x, landmark.y, landmark.z])
    return np.array(coords).reshape(1, -1)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("=" * 50)
print("HAND GESTURE RECOGNITION (Custom Model)")
print("=" * 50)
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = recognizer.recognize(mp_image)

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        draw_hand_landmarks(frame, hand_landmarks)

        X = extract_landmarks(hand_landmarks)
        prediction = model.predict(X)
        gesture = label_encoder.inverse_transform(prediction)[0]

        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
