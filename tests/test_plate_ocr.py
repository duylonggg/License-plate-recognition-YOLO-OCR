"""
Unit tests for ocr/plate_ocr.py

Tests cover preprocessing helpers and the normalize_plate_text function without
requiring EasyOCR to be installed (OCR tests are skipped when easyocr is absent).

Run with:
    pytest tests/test_plate_ocr.py -v
"""

import importlib
import os
import tempfile

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plate_image(text: str = "30A12345", w: int = 200, h: int = 50) -> np.ndarray:
    """Create a simple synthetic license-plate image (white text on black bg)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(
        img, text,
        (5, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (255, 255, 255), 2,
        cv2.LINE_AA,
    )
    return img


# ---------------------------------------------------------------------------
# Import under test (OCR functions that don't need easyocr)
# ---------------------------------------------------------------------------

from ocr.plate_ocr import (
    preprocess_plate,
    normalize_plate_text,
    read_license_plate,
    _to_grayscale,
    _resize,
    _enhance_contrast,
    _denoise,
    _threshold,
    _deskew,
    _apply_positional_corrections,
)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

class TestToGrayscale:
    def test_bgr_to_gray(self):
        img = np.ones((10, 20, 3), dtype=np.uint8) * 128
        result = _to_grayscale(img)
        assert result.ndim == 2

    def test_already_gray(self):
        img = np.ones((10, 20), dtype=np.uint8) * 200
        result = _to_grayscale(img)
        assert result.ndim == 2
        np.testing.assert_array_equal(result, img)

    def test_bgra_to_gray(self):
        img = np.ones((10, 20, 4), dtype=np.uint8) * 100
        result = _to_grayscale(img)
        assert result.ndim == 2


class TestResize:
    def test_upscales_small_image(self):
        img = np.zeros((20, 60), dtype=np.uint8)
        result = _resize(img, target_height=64)
        assert result.shape[0] == 64

    def test_preserves_aspect_ratio(self):
        img = np.zeros((20, 60), dtype=np.uint8)
        result = _resize(img, target_height=64)
        # aspect ratio: 60/20 = 3.0 → expected width ≈ 192
        assert abs(result.shape[1] / result.shape[0] - 3.0) < 0.1

    def test_no_downscale(self):
        img = np.zeros((100, 300), dtype=np.uint8)
        result = _resize(img, target_height=64)
        assert result.shape[0] == 100  # already taller than target – unchanged


class TestEnhanceContrast:
    def test_output_same_shape(self):
        img = np.random.randint(0, 256, (64, 200), dtype=np.uint8)
        result = _enhance_contrast(img)
        assert result.shape == img.shape

    def test_output_dtype(self):
        img = np.random.randint(0, 256, (64, 200), dtype=np.uint8)
        result = _enhance_contrast(img)
        assert result.dtype == np.uint8


class TestDenoise:
    def test_output_same_shape(self):
        img = np.random.randint(0, 256, (64, 200), dtype=np.uint8)
        result = _denoise(img)
        assert result.shape == img.shape


class TestThreshold:
    def test_binary_output(self):
        img = np.random.randint(0, 256, (64, 200), dtype=np.uint8)
        result = _threshold(img)
        unique_vals = set(result.flatten().tolist())
        assert unique_vals.issubset({0, 255})


class TestDeskew:
    def test_no_change_on_blank_image(self):
        img = np.zeros((64, 200), dtype=np.uint8)
        result = _deskew(img)
        assert result.shape == img.shape

    def test_output_same_shape(self):
        img = np.random.randint(0, 256, (64, 200), dtype=np.uint8)
        result = _deskew(img)
        assert result.shape == img.shape


class TestPreprocessPlate:
    def test_output_is_grayscale(self):
        img = _make_plate_image()
        result = preprocess_plate(img)
        assert result.ndim == 2

    def test_output_dtype(self):
        img = _make_plate_image()
        result = preprocess_plate(img)
        assert result.dtype == np.uint8

    def test_output_is_binary(self):
        img = _make_plate_image()
        result = preprocess_plate(img)
        unique_vals = set(result.flatten().tolist())
        assert unique_vals.issubset({0, 255})


# ---------------------------------------------------------------------------
# Positional corrections
# ---------------------------------------------------------------------------

class TestApplyPositionalCorrections:
    def test_province_digits_corrected(self):
        # "OO" at the start should become "00"
        result = _apply_positional_corrections("OOA12345")
        assert result[:2] == "00"

    def test_series_letters_preserved(self):
        result = _apply_positional_corrections("30A12345")
        assert result[2] == "A"

    def test_sequence_digits_corrected(self):
        # "I" in digit sequence should become "1"
        result = _apply_positional_corrections("30AI234S")
        # position 3 onward: "234S" → last char S → 5
        assert result[-1] == "5"

    def test_full_plate_no_errors(self):
        result = _apply_positional_corrections("30A12345")
        assert result == "30A12345"


# ---------------------------------------------------------------------------
# normalize_plate_text
# ---------------------------------------------------------------------------

class TestNormalizePlateText:
    @pytest.mark.parametrize("raw, expected", [
        ("30A12345", "30A12345"),          # clean input
        ("30a12345", "30A12345"),          # lowercase → uppercase
        (" 30A 12345 ", "30A12345"),       # strip spaces
        ("30A-12345", "30A12345"),         # remove hyphen
        ("OOA12345", "00A12345"),          # province code letter→digit
        ("30A1234S", "30A12345"),          # sequence S→5
        ("", ""),                          # empty input
        ("???", ""),                       # only punctuation
    ])
    def test_normalize(self, raw, expected):
        assert normalize_plate_text(raw) == expected

    def test_returns_string(self):
        assert isinstance(normalize_plate_text("30A12345"), str)

    def test_non_matching_format_still_returned(self):
        # Partial / unusual plates should not be discarded.
        result = normalize_plate_text("ABCD")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# read_license_plate – file-system integration (no easyocr required)
# ---------------------------------------------------------------------------

class TestReadLicensePlate:
    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_license_plate("/tmp/nonexistent_plate_image_xyz.jpg")

    def test_raises_value_error_on_bad_file(self, tmp_path):
        bad_file = tmp_path / "bad.jpg"
        bad_file.write_text("this is not an image")
        with pytest.raises(ValueError):
            read_license_plate(str(bad_file))

    @pytest.mark.skipif(
        importlib.util.find_spec("easyocr") is None,
        reason="easyocr not installed",
    )
    def test_returns_string_on_valid_image(self, tmp_path):
        img = _make_plate_image("30A12345")
        path = str(tmp_path / "plate.jpg")
        cv2.imwrite(path, img)
        result = read_license_plate(path)
        assert isinstance(result, str)
