# Gesture Recognition Web Application

Real-time hand gesture recognition with a modern web interface.

## Features

- Real-time webcam streaming
- Hand landmark detection and visualization
- Multiple gesture recognition support
- Configurable settings (labels, landmarks, image quality)
- FPS monitoring
- Responsive dark theme UI

## Setup

```bash
# From project root
cd computer_vision_app

# Run with uv
uv run python app.py

# Or activate virtual environment first
source ../.venv/bin/activate  # Linux/Mac
../.venv/Scripts/activate     # Windows
python app.py
```

## Access

Open your browser to: http://localhost:5001

## How It Works

1. Client streams webcam frames via WebSocket
2. Server processes frames using MediaPipe
3. Hand landmarks are extracted and gestures recognized
4. Processed image and gesture data sent back to client
5. Client renders visualization in real-time

## Configuration

- **Show Labels** - Display gesture names on video
- **Show Landmarks** - Render hand skeleton overlay
- **Image Quality** - Adjust JPEG compression (10-100%)

## Architecture

```
computer_vision_app/
├── app.py              # FastHTML application + WebSocket handler
├── core/
│   ├── gesture_recognition.py  # Main recognition logic
│   ├── model_loader.py         # Model initialization
│   ├── image_utils.py          # Image encoding/decoding
│   └── webcam_recog.py         # Webcam utilities
└── main.py             # Entry point
```

## Models

Requires MediaPipe gesture recognition model:
- `models/gesture_recognizer.task`
- Download from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer)
