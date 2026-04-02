import cv2
import numpy as np
import base64


def decode_image(data_url):
    try:
        _, encoded = data_url.split(",", 1)
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        return None


def encode_image(img, imageQuality):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, imageQuality])
    encoded = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"
