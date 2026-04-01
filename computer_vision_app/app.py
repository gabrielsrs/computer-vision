import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, VideoHTMLAttributes
import av
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_loader import check_models_exist, load_models
from core.gesture_recognition import recognize_gesture


class GestureVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.models = None
        try:
            if check_models_exist():
                self.models = load_models()
                self.ready = True
            else:
                self.ready = False
        except Exception:
            self.ready = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if self.ready and self.models:
            img = recognize_gesture(img, self.models)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("Gesture Recognition")

if not check_models_exist():
    st.error(
        "Models not found. Please ensure model files exist in the 'models/' directory."
    )
else:
    st.info("Camera is active. Show your hand gestures!")

    webrtc_streamer(
        key="gesture-recognition",
        video_processor_factory=GestureVideoProcessor,
        media_stream_constraints={
            "video": {"width": 640, "height": 480},
            "audio": False,
        },
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=True),
    )
