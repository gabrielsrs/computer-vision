from fasthtml.common import *
import cv2
import numpy as np
import base64

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
from core.model_loader import load_models
from core.gesture_recognition import recognize_gesture
from core.image_utils import decode_image, encode_image

app, rt = fast_app(live=True, static_path=os.path.join(BASE_DIR, "assets"))
models = load_models()


@rt("/")
def get():
    return Html(
        Head(
            Title("Gesture Recognition"),
            Meta(charset="utf-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            Style("""
                body { font-family: sans-serif; text-align: center; padding: 20px; }
                canvas { border: 2px solid #333; margin: 10px; }
                .status { padding: 10px; border-radius: 5px; margin-bottom: 15px; }
                .error { background: #ffcccc; color: #990000; }
                .success { background: #ccffcc; color: #009900; }
            """),
        ),
        Body(
            H1("Gesture Recognition"),
            Main(
                Video(
                    id="video",
                    autoplay=True,
                    playsinline=True,
                    style="displaay: none",
                ),
                Canvas(id="canvas"),
                Script(src="main.js"),
            ),
        ),
    )


@app.ws("/ws")
async def ws_endpoint(image: str, send):
    img = decode_image(image)
    if img is not None:
        processed_img = recognize_gesture(img, models)
        await send(encode_image(processed_img))


serve()
