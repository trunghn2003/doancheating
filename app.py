"""
Flask entry-point exposing the cheating detection pipeline as a REST API.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict

import cv2
from flask import Flask, jsonify, request, url_for

from cheating_detection import annotate_detections, load_default_pipeline
from cheating_detection.utils import (
    decode_image_from_base64,
    decode_image_from_bytes,
)

LOGGER = logging.getLogger(__name__)

app = Flask(__name__)
PIPELINE = load_default_pipeline()
ANNOTATED_DIR = Path(app.static_folder) / "annotated"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok"})


@app.route("/api/detect", methods=["POST"])
def detect() -> Any:

    payload = request.get_json(silent=True) if request.is_json else None
    try:
        image = _extract_image_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        result = PIPELINE.analyze(image)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Cheating detection failed")
        return jsonify({"error": "Internal detection failure"}), 500

    try:
        annotated = annotate_detections(image, result)
        filename = _persist_annotated_image(annotated)
        result["annotated_image_url"] = url_for(
            "static", filename=f"annotated/{filename}", _external=True
        )
    except ValueError as exc:
        LOGGER.warning("Failed to persist annotated image: %s", exc)

    return jsonify(result)


def _extract_image_payload(payload: Dict[str, Any] | None = None):
    """
    Attempt to load an image from the incoming request.
    """
    if request.files:
        file_storage = request.files.get("file")
        if not file_storage or not file_storage.filename:
            raise ValueError("Missing uploaded file")
        return decode_image_from_bytes(file_storage.read())

    if payload is None and request.is_json:
        payload = request.get_json() or {}
    if payload:
        if "image_base64" in payload:
            return decode_image_from_base64(payload["image_base64"])
        if "image_bytes" in payload:
            return decode_image_from_base64(payload["image_bytes"])

    raise ValueError(
        "Unsupported request payload. Provide 'file' via form-data "
        "or 'image_base64' within a JSON body."
    )


@app.route("/api/faces", methods=["POST"])
def add_face() -> Any:
    """
    Register a new identity in the face database.
    """
    try:
        name = _extract_name()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    images = _extract_images()
    if not images:
        return jsonify({"error": "At least one image is required"}), 400

    try:
        summary = PIPELINE.face_recognizer.add_person(name, images)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:  # pragma: no cover - defensive
        LOGGER.exception("Failed to add identity")
        return jsonify({"error": "Internal error while adding face"}), 500

    summary["total_identities"] = len(PIPELINE.face_recognizer.database.people)
    return jsonify(summary), 201


def _extract_name() -> str:
    if request.is_json:
        payload = request.get_json() or {}
        name = payload.get("name")
        if name:
            return name
    if request.form:
        name = request.form.get("name")
        if name:
            return name
    raise ValueError("Missing 'name' field")


def _extract_images():
    images = []
    if request.files:
        file_list = request.files.getlist("images") or request.files.getlist("file")
        for item in file_list:
            if not item or not item.filename:
                continue
            try:
                images.append(decode_image_from_bytes(item.read()))
            except ValueError as exc:
                LOGGER.warning("Failed to decode uploaded image: %s", exc)
    elif request.is_json:
        payload: Dict[str, Any] = request.get_json() or {}
        base64_images = payload.get("images")
        single = payload.get("image_base64") or payload.get("image_bytes")
        if single and not base64_images:
            base64_images = [single]
        if base64_images:
            for data in base64_images:
                try:
                    images.append(decode_image_from_base64(data))
                except ValueError as exc:
                    LOGGER.warning("Failed to decode base64 image: %s", exc)
    return images


def _persist_annotated_image(image) -> str:
    """
    Store the annotated frame on disk and return the relative filename.
    """
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Failed to encode annotated frame")
    filename = f"{uuid.uuid4().hex}.jpg"
    output_path = ANNOTATED_DIR / filename
    output_path.write_bytes(buffer.tobytes())
    return filename


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=8000, debug=False)
