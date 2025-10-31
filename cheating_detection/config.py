"""
Centralised configuration helpers for the cheating detection service.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent.parent

# Default artefact locations constructed relative to the repository root.
DEFAULT_FACE_DATASET_DIR = (
    BASE_DIR / "Face_Recognition_Training" / "FaceDataset"
)
DEFAULT_YOLO_MODEL_PATH = BASE_DIR / "Object_detect" / "best.pt"

# Default head pose thresholds expressed in degrees.
DEFAULT_HEAD_POSE_THRESHOLDS: Dict[str, float] = {
    "yaw": 20.0,
    "pitch": 20.0,
    "roll": 20.0,
}


def resolve_path(path_like: Any) -> Path:
    """
    Convert a configuration value to a pathlib.Path, expanding user and vars.
    """
    if isinstance(path_like, Path):
        return path_like
    return Path(str(path_like)).expanduser().resolve()
