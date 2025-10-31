"""
High-level package exports for the cheating detection system.
"""

from .face_database import FaceDatabase
from .face_recognition import FaceRecognizer
from .gaze import EyeGazeEstimator
from .pipeline import CheatingDetectionPipeline, load_default_pipeline
from .visualization import annotate_detections

__all__ = [
    "CheatingDetectionPipeline",
    "FaceDatabase",
    "FaceRecognizer",
    "EyeGazeEstimator",
    "annotate_detections",
    "load_default_pipeline",
]
