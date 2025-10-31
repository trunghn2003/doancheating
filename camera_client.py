"""
Capture frames from the default camera, send them to the Flask API, and display annotated results.
"""

from __future__ import annotations

import base64
import time
from typing import Tuple

import cv2
import numpy as np
import requests


API_URL = "http://localhost:8000/api/detect?return_image=true"
WINDOW_TITLE = "Cheating Detection"


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera (device 0)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            annotated, latency, response = process_frame(frame)
            if annotated is None:
                display_frame = frame
            else:
                display_frame = annotated

            overlay_status(display_frame, latency, response)
            cv2.imshow(WINDOW_TITLE, display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def process_frame(frame: np.ndarray) -> Tuple[np.ndarray | None, float, dict]:
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return None, 0.0, {}

    files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
    start = time.time()
    response = requests.post(API_URL, files=files, timeout=10)
    latency = time.time() - start
    response.raise_for_status()
    data = response.json()

    annotated = None
    encoded = data.get("annotated_image_base64")
    if encoded:
        annotated = decode_base64_image(encoded)
    return annotated, latency, data


def decode_base64_image(encoded: str) -> np.ndarray | None:
    try:
        binary = base64.b64decode(encoded)
    except (ValueError, TypeError):
        return None
    array = np.frombuffer(binary, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def overlay_status(frame: np.ndarray, latency: float, response: dict) -> None:
    text = f"latency: {latency*1000:.0f} ms"
    flags = response.get("flags") or []
    status = response.get("status", "n/a")
    cv2.putText(
        frame,
        f"{status.upper()} | {text}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{status.upper()} | {text}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for idx, flag in enumerate(flags[:3], start=1):
        y = 20 + idx * 20
        cv2.putText(
            frame,
            flag,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            flag,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    main()
