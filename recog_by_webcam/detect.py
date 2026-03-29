import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

# Create the object detector instance with local EfficientDet-Lite0 model
# base_options specifies the TFLite model file to use
# max_results=5 limits the number of detected objects to display
base_options = mp.tasks.BaseOptions(model_asset_path="models/efficientdet_lite0.tflite")
options = vision.ObjectDetectorOptions(base_options=base_options, max_results=5)
object_detector = vision.ObjectDetector.create_from_options(options)


def draw_detection(frame, detection):
    """Draw bounding box and label for a single detection"""
    # Get bounding box coordinates (normalized 0-1 values)
    bbox = detection.bounding_box
    category = detection.categories[0]

    # Scale bbox to frame dimensions
    x = bbox.origin_x
    y = bbox.origin_y
    w = bbox.width
    h = bbox.height

    # Draw rectangle around detected object
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Prepare label text with category name and confidence
    label = f"{category.category_name} ({category.score:.2f})"

    # Draw label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        frame, (x, y - label_size[1] - 10), (x + label_size[0], y), (0, 255, 0), -1
    )

    # Draw label text
    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


# Open the default webcam (index 0)
cap = cv2.VideoCapture(1)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Press 'q' to quit")

# Main loop to capture and process frames
while True:
    # Capture frame-by-frame from webcam
    # ret indicates if frame was captured successfully
    # frame contains the image data
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Convert BGR frame to RGB (MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image object from the RGB frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Process the frame to detect objects
    results = object_detector.detect(mp_image)

    # Draw bounding boxes and labels for detected objects
    if results.detections:
        for detection in results.detections:
            draw_detection(frame, detection)

    # Display the resulting frame with detections
    cv2.imshow("Object Detection", frame)

    # Exit loop when 'q' key is pressed
    # waitKey(1) returns ASCII code of pressed key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
