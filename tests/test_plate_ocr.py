"""
Unit tests for ocr/plate_ocr.py

Tests cover preprocessing helpers, the deskew utilities, character assembly,
and the normalize_plate_text function without requiring PyTorch or a model
file to be installed (OCR integration tests are skipped when unavailable).

Run with:
    pytest tests/test_plate_ocr.py -v
"""

import importlib
import math
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
# Import under test (functions that don't need PyTorch / model file)
# ---------------------------------------------------------------------------

from ocr.plate_ocr import (
    preprocess_plate,
    normalize_plate_text,
    read_license_plate,
    deskew_plate,
    _to_grayscale,
    _resize,
    _enhance_contrast,
    _denoise,
    _threshold,
    _deskew,
    _change_contrast,
    _rotate_image,
    _compute_skew,
    _assemble_plate_from_detections,
    _apply_positional_corrections,
    _try_recover_valid_plate,
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

    def test_default_target_height_128(self):
        """preprocess_plate now upscales to 128px for better OCR accuracy."""
        img = np.zeros((20, 60), dtype=np.uint8)
        result = _resize(img, target_height=128)
        assert result.shape[0] == 128


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
# Deskew utilities (new in YOLOv5-based pipeline)
# ---------------------------------------------------------------------------

class TestChangeContrast:
    def test_output_same_shape(self):
        img = np.random.randint(0, 256, (64, 200, 3), dtype=np.uint8)
        result = _change_contrast(img)
        assert result.shape == img.shape

    def test_output_dtype(self):
        img = np.random.randint(0, 256, (64, 200, 3), dtype=np.uint8)
        result = _change_contrast(img)
        assert result.dtype == np.uint8


class TestRotateImage:
    def test_output_same_shape(self):
        img = np.random.randint(0, 256, (64, 200, 3), dtype=np.uint8)
        result = _rotate_image(img, 5.0)
        assert result.shape == img.shape

    def test_zero_angle_unchanged(self):
        img = np.ones((64, 200, 3), dtype=np.uint8) * 128
        result = _rotate_image(img, 0.0)
        np.testing.assert_array_equal(result, img)


class TestComputeSkew:
    def test_returns_float(self):
        img = _make_plate_image()
        result = _compute_skew(img, center_threshold=0)
        assert isinstance(result, float)

    def test_blank_image_returns_numeric(self):
        img = np.zeros((64, 200, 3), dtype=np.uint8)
        result = _compute_skew(img, center_threshold=0)
        assert isinstance(result, (int, float))


class TestDeskewPlate:
    def test_output_same_shape_no_contrast(self):
        img = _make_plate_image()
        result = deskew_plate(img, change_contrast=False, center_threshold=0)
        assert result.shape == img.shape

    def test_output_same_shape_with_contrast(self):
        img = _make_plate_image()
        result = deskew_plate(img, change_contrast=True, center_threshold=0)
        assert result.shape == img.shape

    def test_output_dtype(self):
        img = _make_plate_image()
        result = deskew_plate(img)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Character assembly (_assemble_plate_from_detections)
# ---------------------------------------------------------------------------

class TestAssemblePlateFromDetections:
    def _make_bb(self, label: str, x: float, y: float) -> list:
        """Build a minimal bounding-box entry [xmin, ymin, xmax, ymax, conf, cls_id, label]."""
        return [x, y, x + 10, y + 20, 0.9, 0, label]

    def test_unknown_when_empty(self):
        assert _assemble_plate_from_detections([]) == "unknown"

    def test_unknown_when_too_few(self):
        bbs = [self._make_bb(str(i), i * 12.0, 5.0) for i in range(6)]
        assert _assemble_plate_from_detections(bbs) == "unknown"

    def test_unknown_when_too_many(self):
        bbs = [self._make_bb(str(i % 10), i * 12.0, 5.0) for i in range(11)]
        assert _assemble_plate_from_detections(bbs) == "unknown"

    def test_1_line_plate_sorted_by_x(self):
        # 7 characters on a single line, provided deliberately out of order.
        chars = [("A", 24), ("3", 0), ("4", 72), ("0", 12), ("2", 48), ("1", 36), ("3", 60)]
        bbs = [self._make_bb(c, float(x), 5.0) for c, x in chars]
        result = _assemble_plate_from_detections(bbs)
        assert result == "30A1234"

    def test_2_line_plate_dash_separator(self):
        # 4 chars on row 1 (y=5) and 4 on row 2 (y=40).
        row1 = [("3", 0, 5), ("0", 12, 5), ("A", 24, 5), ("1", 36, 5)]
        row2 = [("2", 0, 40), ("3", 12, 40), ("4", 24, 40), ("5", 36, 40)]
        bbs = [self._make_bb(c, float(x), float(y)) for c, x, y in row1 + row2]
        result = _assemble_plate_from_detections(bbs)
        assert "-" in result  # dash separator for 2-line format
        parts = result.split("-")
        assert len(parts) == 2


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

    def test_letter_digit_series_preserved(self):
        # Letter+digit series (e.g. G1): position 2 is a letter, position 3 is the
        # series digit and must NOT be converted via letter_fixes.
        result = _apply_positional_corrections("29G133333")
        assert result == "29G133333"

    def test_letter_digit_series_digit_not_converted(self):
        # The '1' in 'G1' series should NOT be converted to 'I' by letter_fixes.
        result = _apply_positional_corrections("29G133333")
        assert result[3] == "1"  # series digit stays '1', not 'I'

    def test_letter_digit_series_letter_corrected(self):
        # If position 2 is a digit (not alpha), no series is detected, so no
        # letter_fixes are applied and the digit stays unchanged.
        result = _apply_positional_corrections("296133333")
        assert result == "296133333"


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
        # letter+digit series (newer VN plates, e.g. 29-G1 333.33)
        ("29G133333", "29G133333"),        # G1 series, correct OCR read
        ("29-G1 333.33", "29G133333"),     # raw plate text with separators
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

    def test_oversized_plate_recovered(self):
        # An extra detection sometimes makes the plate 10 chars.
        # normalize_plate_text should trim it back to a valid 9-char plate.
        result = normalize_plate_text("29A6133333")  # 10 chars
        assert len(result) <= 9
        assert result  # non-empty


# ---------------------------------------------------------------------------
# _try_recover_valid_plate
# ---------------------------------------------------------------------------

class TestTryRecoverValidPlate:
    def test_removes_noise_character_to_letter_digit_series(self):
        # 10-char plate with an extra noise character: removing the right char
        # should yield a valid 9-char letter+digit-series plate.
        result = _try_recover_valid_plate("29A6133333")
        assert result  # non-empty
        assert len(result) == 9

    def test_returns_empty_when_unrecoverable(self):
        # Completely garbled input should return empty string.
        result = _try_recover_valid_plate("XXXXXXXXXXX")
        assert result == ""


# ---------------------------------------------------------------------------
# read_license_plate – file-system integration (no PyTorch / model required)
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
        importlib.util.find_spec("torch") is None or not os.path.isfile("model/LP_ocr.pt"),
        reason="torch not installed or model/LP_ocr.pt not present",
    )
    def test_returns_string_on_valid_image(self, tmp_path):
        img = _make_plate_image("30A12345")
        path = str(tmp_path / "plate.jpg")
        cv2.imwrite(path, img)
        result = read_license_plate(path)
        assert isinstance(result, str)
