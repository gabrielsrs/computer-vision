import cv2
import mediapipe as mp

# Create the gesture recognizer instance with local model
base_options = mp.tasks.BaseOptions(model_asset_path="models/gesture_recognizer.task")
VisionRunningMode = mp.tasks.vision.RunningMode
options = mp.tasks.vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=VisionRunningMode.IMAGE,
)
recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

# Hand landmark connections (MediaPipe standard hand connections)
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


def draw_hand_landmarks(frame, hand_landmarks):
    """Draw hand landmarks and connections on the frame"""
    h, w = frame.shape[:2]

    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    # Draw landmark points
    for idx, landmark in enumerate(hand_landmarks):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        # Wrist is green, other points are red
        color = (0, 255, 0) if idx == 0 else (0, 0, 255)
        cv2.circle(frame, (x, y), 5, color, -1)


# Open the default webcam (index 0)
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Press 'q' to quit")

# Main loop to capture and process frames
while True:
    # Capture frame-by-frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Convert BGR frame to RGB (MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image object from the RGB frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Process the frame to recognize gestures
    result = recognizer.recognize(mp_image)

    # Draw hand landmarks and display gesture labels
    if result.hand_landmarks:
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            draw_hand_landmarks(frame, hand_landmarks)

            # Display gesture label at hand position
            if idx < len(result.gestures):
                gesture = result.gestures[idx]
                gesture_name = gesture[0].category_name
                confidence = gesture[0].score

                # Get hand wrist position for label placement
                wrist = hand_landmarks[0]
                x = int(wrist.x * frame.shape[1])
                y = int(wrist.y * frame.shape[0]) - 30

                label = f"{gesture_name} ({confidence:.2f})"
                cv2.putText(
                    frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                print(
                    f"Detected gesture: {gesture_name} (confidence: {confidence:.2f})"
                )

    # Display the resulting frame with landmarks and gestures
    cv2.imshow("Gesture Recognition", frame)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
