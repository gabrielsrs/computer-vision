from fasthtml.common import *
import cv2
import numpy as np
import base64
import json

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
            Link(rel="preconnect", href="https://fonts.googleapis.com"),
            Link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin=""),
            Link(
                href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap",
                rel="stylesheet",
            ),
            Style("""
                * { box-sizing: border-box; margin: 0; padding: 0; }
                body { 
                    font-family: 'Poppins', sans-serif; 
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 40px 20px;
                    color: #fff;
                }
                h1 { 
                    font-size: 2.5rem; 
                    font-weight: 700; 
                    margin-bottom: 30px;
                    background: linear-gradient(90deg, #00d9ff, #00ff88);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    text-shadow: 0 0 30px rgba(0, 217, 255, 0.3);
                }
                .container {
                    display: flex;
                    gap: 30px;
                    flex-wrap: wrap;
                    justify-content: center;
                    max-width: 1200px;
                }
                .video-section {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 20px;
                    padding: 20px;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                }
                #canvas { 
                    border-radius: 15px;
                    display: block;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
                }
                .gesture-panel {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 20px;
                    padding: 25px;
                    min-width: 280px;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                }
                .panel-title {
                    font-size: 1.2rem;
                    font-weight: 600;
                    margin-bottom: 20px;
                    color: #00d9ff;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .gesture-badge {
                    display: inline-flex;
                    align-items: center;
                    gap: 10px;
                    padding: 12px 18px;
                    background: rgba(0, 255, 136, 0.1);
                    border: 1px solid rgba(0, 255, 136, 0.3);
                    border-radius: 12px;
                    margin-bottom: 10px;
                    animation: fadeIn 0.3s ease;
                }
                .gesture-badge.matched {
                    background: rgba(0, 255, 136, 0.2);
                    border-color: #00ff88;
                    box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
                }
                .gesture-name {
                    font-weight: 600;
                    font-size: 1.1rem;
                    text-transform: capitalize;
                }
                .gesture-prob {
                    font-size: 0.85rem;
                    color: rgba(255, 255, 255, 0.7);
                }
                .hand-label {
                    font-size: 0.75rem;
                    padding: 4px 10px;
                    background: rgba(0, 217, 255, 0.2);
                    border-radius: 20px;
                    color: #00d9ff;
                }
                .matched-gesture {
                    text-align: center;
                    padding: 30px;
                    background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 217, 255, 0.1));
                    border-radius: 15px;
                    border: 2px solid #00ff88;
                    margin-top: 20px;
                    animation: pulse 2s infinite;
                }
                .matched-gesture img {
                    width: 120px;
                    height: 120px;
                    object-fit: contain;
                    margin-bottom: 15px;
                    filter: drop-shadow(0 0 20px rgba(0, 255, 136, 0.5));
                }
                .matched-title {
                    font-size: 1.5rem;
                    font-weight: 700;
                    color: #00ff88;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                }
                .empty-state {
                    color: rgba(255, 255, 255, 0.5);
                    font-style: italic;
                    text-align: center;
                    padding: 40px;
                }
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(-10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                @keyframes pulse {
                    0%, 100% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.3); }
                    50% { box-shadow: 0 0 40px rgba(0, 255, 136, 0.6); }
                }
                .loading {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 10px;
                    color: rgba(255, 255, 255, 0.7);
                }
                .spinner {
                    width: 20px;
                    height: 20px;
                    border: 2px solid rgba(255, 255, 255, 0.3);
                    border-top-color: #00d9ff;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                }
                @keyframes spin { to { transform: rotate(360deg); } }
                .config-panel {
                    display: flex;
                    flex-direction: column;
                    width: 100%;
                    gap: 20px;
                }
                .config-group {
                    display: flex;
                }
                .config-panel {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 16px;
                    padding: 20px;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    margin-bottom: 25px;
                    max-width: 600px;
                    width: 100%;
                }
                .config-title {
                    font-size: 1rem;
                    font-weight: 600;
                    margin-bottom: 15px;
                    color: #00d9ff;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                .config-group {
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                    margin-bottom: 15px;
                    padding-bottom: 15px;
                }
                .config-group:last-child {
                    border-bottom: none;
                    margin-bottom: 0;
                    padding-bottom: 0;
                }
                .config-label {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    color: rgba(255, 255, 255, 0.85);
                    font-size: 0.9rem;
                    cursor: pointer;
                    transition: color 0.2s ease;
                }
                .config-label:hover {
                    color: #00d9ff;
                }
                .config-label input[type="checkbox"] {
                    appearance: none;
                    width: 20px;
                    height: 20px;
                    border: 2px solid rgba(255, 255, 255, 0.3);
                    border-radius: 6px;
                    background: rgba(255, 255, 255, 0.05);
                    cursor: pointer;
                    position: relative;
                    transition: all 0.2s ease;
                }
                .config-label input[type="checkbox"]:checked {
                    background: linear-gradient(135deg, #00d9ff, #00ff88);
                    border-color: #00ff88;
                }
                .config-label input[type="checkbox"]:checked::after {
                    content: "✓";
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    color: #1a1a2e;
                    font-size: 12px;
                    font-weight: 700;
                }
                .config-label input[type="range"] {
                    width: 100%;
                    height: 6px;
                    border-radius: 3px;
                    background: rgba(255, 255, 255, 0.1);
                    appearance: none;
                    cursor: pointer;
                }
                .config-label input[type="range"]::-webkit-slider-thumb {
                    appearance: none;
                    width: 18px;
                    height: 18px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #00d9ff, #00ff88);
                    cursor: pointer;
                    box-shadow: 0 2px 10px rgba(0, 217, 255, 0.4);
                    transition: transform 0.2s ease;
                }
                .config-label input[type="range"]::-webkit-slider-thumb:hover {
                    transform: scale(1.1);
                }
                .quality-display {
                    display: flex;
                    justify-content: space-between;
                    font-size: 0.8rem;
                    color: rgba(255, 255, 255, 0.6);
                }
                #fps {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    position: absolute;
                    top: 32px;
                    right: 32px;
                    background: rgba(0, 0, 0, 0.6);
                    padding: 8px 14px;
                    border-radius: 20px;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(0, 217, 255, 0.3);
                }
                #fps .fps-label {
                    font-size: 0.75rem;
                    font-weight: 600;
                    color: rgba(255, 255, 255, 0.7);
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }
                .fps-display {
                    font-size: 1rem;
                    font-weight: 700;
                    color: #00ff88;
                    text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
                }
            """),
        ),
        Body(
            H1("Gesture Recognition"),
            Div(
                Div(
                    P(Span("⚙️"), Span("Settings"), cls="config-title"),
                    Div(
                        Div(
                            Label(
                                Input(
                                    type="checkbox",
                                    id="enable-landmarks",
                                    checked=True,
                                ),
                                " Show Landmarks",
                                cls="config-label",
                            ),
                            cls="config-group",
                        ),
                        Div(
                            Label(
                                Input(
                                    type="checkbox",
                                    id="enable-gestures",
                                    checked=True,
                                ),
                                " Show Gestures",
                                cls="config-label",
                            ),
                            cls="config-group",
                        ),
                        Div(
                            Div(
                                Label("Image Quality:", cls="config-label"),
                                Div(
                                    Span("60%", id="quality-value"),
                                    Span("", id="resolution-display"),
                                    cls="quality-display",
                                ),
                                style="display: flex; align-items: center; gap: 4px;"
                            ),
                            Input(
                                type="range",
                                id="image-quality",
                                min="10",
                                max="100",
                                value="60",
                            ),
                            cls="config-group",
                        ),
                        style="display: flex; gap: 20px;",
                    ),
                    cls="config-panel",
                ),
            ),
            Div(
                Div(
                    Div(
                        Canvas(id="canvas"),
                    ),
                    Div(
                        Span("FPS", cls="fps-label"),
                        Div("0", id="fps-display", cls="fps-display"),
                        id="fps",
                    ),
                    cls="video-section",
                ),
                Div(
                    Div(
                        "👋 Detected Gestures",
                        cls="panel-title",
                    ),
                    Div(
                        "No hands detected yet...",
                        cls="empty-state",
                    ),
                    id="gesture-container",
                    cls="gesture-panel",
                ),
                cls="container",
            ),
            Video(
                id="video",
                autoplay=True,
                playsinline=True,
                style="display: none",
            ),
            Script(src="main.js"),
        ),
    )


@app.ws("/ws")
async def ws_endpoint(image: str, send):
    img = decode_image(image)
    if img is not None:
        processed_img, gesture_data, matched_gesture = recognize_gesture(img, models)
        response = {
            "image": encode_image(processed_img),
            "gestures": gesture_data,
            "matchedGesture": matched_gesture,
        }
        await send(json.dumps(response))


serve()
