"""
Utility helpers for image handling and result formatting.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np


def decode_image_from_bytes(content: bytes) -> np.ndarray:
    """
    Decode raw bytes into an OpenCV BGR image.

    Raises:
        ValueError: When the bytes cannot be decoded into an image.
    """
    array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image payload")
    return image


def decode_image_from_base64(data: str) -> np.ndarray:
    """
    Decode a base64-encoded string into an OpenCV BGR image.
    """
    try:
        binary = base64.b64decode(data)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid base64 image payload") from exc
    return decode_image_from_bytes(binary)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert an OpenCV BGR image to RGB order.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """
    Coerce an image to uint8 for downstream model consumption.
    """
    if image.dtype == np.uint8:
        return image
    clipped = np.clip(image, 0, 255)
    return clipped.astype(np.uint8)


def serialize_bbox(bbox: Iterable[float]) -> List[float]:
    """
    Convert a bounding box iterable to a plain list of floats.
    """
    return [float(x) for x in bbox]


def to_float(value: Any) -> float:
    """
    Convert numeric-like values to primitive float.
    """
    return float(value)
