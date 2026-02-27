"""
OCR package for Vietnamese license plate recognition.

Main entry point: read_license_plate(image_path)
"""

from .plate_ocr import preprocess_plate, run_ocr, normalize_plate_text, read_license_plate

__all__ = [
    "preprocess_plate",
    "run_ocr",
    "normalize_plate_text",
    "read_license_plate",
]
