"""
Flask entry-point exposing the cheating detection pipeline as a REST API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from flask import Flask, jsonify, request

from cheating_detection import load_default_pipeline
from cheating_detection.utils import decode_image_from_base64, decode_image_from_bytes

LOGGER = logging.getLogger(__name__)

app = Flask(__name__)
PIPELINE = load_default_pipeline()


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok"})


@app.route("/api/detect", methods=["POST"])
def detect() -> Any:
    try:
        image = _extract_image_payload()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        result = PIPELINE.analyze(image)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Cheating detection failed")
        return jsonify({"error": "Internal detection failure"}), 500

    return jsonify(result)


def _extract_image_payload():
    """
    Attempt to load an image from the incoming request.
    """
    if request.files:
        file_storage = request.files.get("file")
        if not file_storage or not file_storage.filename:
            raise ValueError("Missing uploaded file")
        return decode_image_from_bytes(file_storage.read())

    if request.is_json:
        payload: Dict[str, Any] = request.get_json() or {}
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=8000, debug=False)
