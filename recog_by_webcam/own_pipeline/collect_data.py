import cv2
import mediapipe as mp
import csv
import os

# Create the gesture recognizer instance with local model
base_options = mp.tasks.BaseOptions(model_asset_path="models/gesture_recognizer.task")
VisionRunningMode = mp.tasks.vision.RunningMode
options = mp.tasks.vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE,
)
recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

# CSV file configuration
CSV_FILE = "trained_data/hand_landmarks_dataset.csv"
LABEL = None  # Set this to your target label (e.g., "thumbs_up", "peace", "fist")

# CSV header - 21 landmarks x 3 coords (x, y, z) + label
HEADER = ["label"]
for i in range(21):
    HEADER.extend([f"x_{i}", f"y_{i}", f"z_{i}"])

# Create CSV file with header if it doesn't exist
file_exists = os.path.exists(CSV_FILE)
csv_file = open(CSV_FILE, mode="a", newline="")
csv_writer = csv.writer(csv_file)
if not file_exists:
    csv_writer.writerow(HEADER)
    csv_file.flush()


def draw_hand_landmarks(frame, hand_landmarks):
    """Draw hand landmarks and connections on the frame"""
    h, w = frame.shape[:2]

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

    for start_idx, end_idx in HAND_CONNECTIONS:
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    for idx, landmark in enumerate(hand_landmarks):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        color = (0, 255, 0) if idx == 0 else (0, 0, 255)
        cv2.circle(frame, (x, y), 5, color, -1)


def extract_landmarks(hand_landmarks):
    """Extract landmark coordinates as a flat list"""
    coords = []
    for landmark in hand_landmarks:
        coords.extend([landmark.x, landmark.y, landmark.z])
    return coords


def save_to_csv(label, hand_landmarks):
    """Save hand landmarks to CSV file"""
    coords = extract_landmarks(hand_landmarks)
    csv_writer.writerow([label] + coords)
    csv_file.flush()


# Open the default webcam (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("=" * 50)
print("HAND LANDMARK DATA COLLECTOR")
print("=" * 50)
print(f"Saving to: {CSV_FILE}")
print()
print("Instructions:")
print("  1. Enter the label name for your gesture")
print("  2. Press 'SPACE' to capture current hand position")
print("  3. Press 'q' to quit")
print("=" * 50)

label = input("Enter gesture label: ").strip()
if not label:
    print("No label entered. Using default: 'unknown'")
    label = "unknown"

print(f"\nRecording for label: '{label}'")
print("Press SPACE to capture, 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = recognizer.recognize(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            draw_hand_landmarks(frame, hand_landmarks)

    cv2.putText(
        frame,
        f"Label: {label} | SPACE: capture | q: quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Hand Landmark Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == 32:  # SPACE key
        if result.hand_landmarks:
            save_to_csv(label, result.hand_landmarks[0])
            print(f"Captured! Total samples for '{label}'")
        else:
            print("No hand detected! Please show your hand.")

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print(f"\nDataset saved to {CSV_FILE}")
