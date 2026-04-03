# Computer Vision - Gesture Recognition

Real-time hand gesture recognition system using MediaPipe and custom ML models.

## Project Structure

```
computer-vision/
├── computer_vision_app/      # Web application (FastHTML + WebSocket)
├── recog_system/             # Notebooks with recognition concept
├── recog_by_webcam/          # Standalone webcam scripts
│   ├── own_pipeline/         # Custom model training pipeline
├── models/                   # Pre-trained models
├── trained_data/             # Training datasets
├── lenet5/                   # LeNet5 MNIST implementation
├── models/                   # Downloaded models
│   ├── own_models/           # Models from trained data
└── pyproject.toml            # Python dependencies
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run the web application
cd computer_vision_app && uv run python app.py

# Or run standalone webcam scripts
python recog_by_webcam/gesture.py       # MediaPipe gesture recognition
python recog_by_webcam/detect.py        # Object detection
```

## Components

### Web Application (`computer_vision_app/`)
Real-time gesture recognition with a web interface.
- FastHTML backend with WebSocket support
- Client-side webcam streaming
- Adjustable image quality and visualization options

See [computer_vision_app/README.md](computer_vision_app/README.md) for details.

### Notebooks with recognition concept (`recog_system/`)
Notebooks with the follow concepts
- Image classification
- Object detection
- Image segmentation

Sourece the notebook [models](https://ai.google.dev/edge/mediapipe/solutions/guide).

### Standalone Scripts (`recog_by_webcam/`)
- `gesture.py` - MediaPipe gesture recognition
- `detect.py` - Object detection with EfficientDet

### Custom Model Pipeline (`recog_by_webcam/own_pipeline/`)
Train your own gesture classifier:
1. `collect_data.py` - Collect hand landmark data
2. `train_model.py` - Train RandomForest classifier
3. `predict.py` - Use trained model for predictions

### LeNet5 (`lenet5/`)
MNIST digit recognition using LeNet5 architecture.

## Requirements

- Python 3.12+
- Webcam for real-time recognition
- MediaPipe models (downloaded to `models/`)

## Dependencies

- mediapipe
- opencv-python
- torch / torchvision
- transformers
- scikit-learn
- joblib
- fasthtml
