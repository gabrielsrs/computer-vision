import cv2
import mediapipe as mp
import numpy as np
import time


def recognize_gesture(image, models, timestamp_ms=None):
    frame = image.copy()
    recognizer = models["recognizer"]
    clf = models["clf"]
    label_encoder = models["label_encoder"]

    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

    if recognition_result.hand_landmarks:
        for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
            mp_hands = mp.tasks.vision.HandLandmarksConnections
            mp_drawing = mp.tasks.vision.drawing_utils
            mp_drawing_styles = mp.tasks.vision.drawing_styles

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            hand_label = recognition_result.handedness[i][0].category_name
            handedness_val = 0 if hand_label == "Left" else 1

            landmarks_array = [handedness_val]
            for lm in hand_landmarks:
                landmarks_array.extend([lm.x, lm.y, lm.z])

            features = np.array(landmarks_array).reshape(1, -1)

            prediction_idx = clf.predict(features)[0]
            prediction_prob = np.max(clf.predict_proba(features))
            gesture_name = label_encoder.inverse_transform([prediction_idx])[0]

            color = (0, 255, 0)

            display_text = (
                f"Custom {hand_label}: {gesture_name} ({prediction_prob:.2f})"
            )
            cv2.putText(
                frame,
                display_text,
                (20, 50 + (i * 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

    return frame
