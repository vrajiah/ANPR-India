"""
ANPR System - Automatic Number Plate Recognition
A production-ready system for Indian license plate detection and recognition.
"""

__version__ = "1.0.0"
__author__ = "ANPR Development Team"
__email__ = "anpr@example.com"

from .core import ANPRSystem, YOLO, PaddleOCR
from .utils import clean_image, filter_text, is_valid_license_plate

__all__ = [
    "ANPRSystem",
    "YOLO", 
    "PaddleOCR",
    "clean_image",
    "filter_text", 
    "is_valid_license_plate",
]
