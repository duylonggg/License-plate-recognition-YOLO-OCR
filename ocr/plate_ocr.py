"""
OCR module for Vietnamese license plate recognition.

Pipeline:
  1. preprocess_plate(image)   – image enhancement before OCR
  2. run_ocr(image)            – EasyOCR-based text extraction
  3. normalize_plate_text(text) – clean and standardise the raw OCR output
  4. read_license_plate(image_path) – end-to-end helper

Installation:
  pip install easyocr opencv-python-headless numpy Pillow

Usage example:
  from ocr.plate_ocr import read_license_plate
  result = read_license_plate("resources/plate.jpg")
  print(result)   # e.g. "30A12345"
"""

import re
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Lazy-load EasyOCR reader so the heavy model is only initialised on first use.
# ---------------------------------------------------------------------------
_reader = None


def _get_reader():
    """Return a cached EasyOCR Reader instance (English only, no GPU by default)."""
    global _reader
    if _reader is None:
        import easyocr  # imported here so the module can be imported without easyocr installed
        # Use English character set; set gpu=True if a CUDA GPU is available.
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


# ---------------------------------------------------------------------------
# Step 1 – Preprocessing
# ---------------------------------------------------------------------------

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR or BGRA image to grayscale."""
    if len(image.shape) == 2:
        return image  # already grayscale
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _resize(image: np.ndarray, target_height: int = 64) -> np.ndarray:
    """
    Upscale the plate image so its height is target_height pixels, preserving
    the aspect ratio.  A taller image gives EasyOCR finer character detail.
    """
    h, w = image.shape[:2]
    if h >= target_height:
        return image
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)


def _enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to boost
    local contrast.  This works well on dirty or unevenly-lit plates.

    Alternative: simple histogram equalization via cv2.equalizeHist(image).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def _denoise(image: np.ndarray) -> np.ndarray:
    """
    Remove noise with a fast median blur.  Kernel size 3 is a good default;
    increase to 5 for noisier images.

    Alternative: cv2.fastNlMeansDenoising(image, h=10) for stronger denoising
    at the cost of speed.
    """
    return cv2.medianBlur(image, 3)


def _threshold(image: np.ndarray) -> np.ndarray:
    """
    Binarise the image with Otsu's thresholding.  Otsu automatically picks the
    optimal global threshold, which works well for two-tone license plates.

    Alternative: adaptive thresholding for plates with uneven lighting –
      cv2.adaptiveThreshold(image, 255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 2)
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _deskew(image: np.ndarray) -> np.ndarray:
    """
    Correct small rotations (skew) in the plate image using the dominant angle
    of detected edges.  Skips correction when the tilt is negligible (<0.5°).

    This step is optional; comment it out in preprocess_plate() if it
    introduces artefacts on your dataset.
    """
    coords = np.column_stack(np.where(image > 0))
    if coords.shape[0] < 5:
        return image  # too few foreground pixels – skip
    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angles in (-90, 0]; map to (-45, 45]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return image  # negligible skew – skip
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def preprocess_plate(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline for a cropped license-plate image.

    Steps applied in order:
      1. Grayscale conversion
      2. Resize (upscale to at least 64 px height)
      3. Contrast enhancement via CLAHE
      4. Noise removal via median blur
      5. Binarisation via Otsu's threshold
      6. Deskew (rotation correction)

    Args:
        image: NumPy array – BGR, BGRA, or grayscale plate image.

    Returns:
        Preprocessed grayscale binary image (NumPy array, dtype uint8).
    """
    img = _to_grayscale(image)
    img = _resize(img, target_height=64)
    img = _enhance_contrast(img)
    img = _denoise(img)
    img = _threshold(img)
    img = _deskew(img)
    return img


# ---------------------------------------------------------------------------
# Step 2 – OCR
# ---------------------------------------------------------------------------

def run_ocr(image: np.ndarray) -> str:
    """
    Run EasyOCR on a preprocessed plate image and return the concatenated raw
    text from all detected regions.

    EasyOCR was chosen over Tesseract and PaddleOCR because:
      - Single pip install, no external binary dependencies.
      - Decent out-of-the-box accuracy for alphanumeric content.
      - Easy to swap GPU/CPU and to extend with other language models.

    Args:
        image: Preprocessed (grayscale/binary) NumPy array.

    Returns:
        Raw OCR string (may contain spaces and noise).
    """
    reader = _get_reader()
    # detail=0 returns plain strings; allowlist restricts to alphanumeric chars.
    results = reader.readtext(
        image,
        detail=0,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        paragraph=False,
    )
    # Join multiple regions (e.g. two-row plates) with a space for downstream handling.
    raw_text = " ".join(results)
    return raw_text


# ---------------------------------------------------------------------------
# Step 3 – Post-processing / normalisation
# ---------------------------------------------------------------------------

# Common single-character OCR confusions for license plate characters.
# Applied after uppercasing so only uppercase variants are needed.
_OCR_CORRECTIONS = {
    "O": "0",  # letter O → digit 0  (position-dependent, see normalize_plate_text)
    "I": "1",  # letter I → digit 1
    "B": "8",  # letter B → digit 8
    "S": "5",  # letter S → digit 5
    "Z": "2",  # letter Z → digit 2
    "G": "6",  # letter G → digit 6
    "T": "7",  # letter T → digit 7  (less common but observed)
}

# Vietnamese license plate format patterns (simplified):
#   Standard (two-row or one-row, 1-letter series): [2 digits][1 letter][4-5 digits]  e.g. 30A12345
#   One-row (2-letter series):                      [2 digits][2 letters][4-5 digits] e.g. 51AB1234
#   Both variants are matched by the regex below ([A-Z]{1,2}).
_VN_PLATE_PATTERN = re.compile(
    r"^(\d{2})([A-Z]{1,2})(\d{4,5})$"
)


def _apply_positional_corrections(plate: str) -> str:
    """
    Apply letter↔digit substitutions based on position in the plate string.

    Vietnamese plates follow: <2-digit province code> <1-2 letter series> <4-5 digit number>

    - Province code (positions 0-1): must be digits → convert look-alike letters.
    - Series (positions 2-3 or 2): must be letters → convert look-alike digits.
    - Sequence (last 4-5 chars): must be digits → convert look-alike letters.
    """
    # Digit look-alikes (used in digit positions)
    digit_fixes = {"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2", "G": "6"}
    # Letter look-alikes (used in letter positions)
    letter_fixes = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}

    chars = list(plate)
    length = len(chars)

    # Province code – first 2 characters must be digits.
    for i in range(min(2, length)):
        chars[i] = digit_fixes.get(chars[i], chars[i])

    # Determine series length heuristically.
    # After the 2-digit province code the series is 1-2 letters.
    series_end = 2
    if length > 2 and chars[2].isalpha():
        series_end = 3
        if length > 3 and chars[3].isalpha():
            series_end = 4

    # Series characters must be letters.
    for i in range(2, series_end):
        chars[i] = letter_fixes.get(chars[i], chars[i])

    # Sequence (after the series) must be digits.
    for i in range(series_end, length):
        chars[i] = digit_fixes.get(chars[i], chars[i])

    return "".join(chars)


def normalize_plate_text(text: str) -> str:
    """
    Clean and standardise raw OCR output into a Vietnamese license plate string.

    Steps:
      1. Uppercase and strip whitespace / non-alphanumeric characters.
      2. Apply positional letter↔digit corrections.
      3. Validate against the expected VN plate format (optional – returns best
         guess even if it does not match, so the caller can decide).

    Args:
        text: Raw OCR string (may contain spaces, punctuation, noise).

    Returns:
        Cleaned plate string, e.g. "30A12345".  Returns an empty string if
        no alphanumeric content could be extracted.
    """
    # 1. Uppercase and keep only alphanumeric characters.
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())

    if not cleaned:
        return ""

    # 2. Apply positional corrections.
    corrected = _apply_positional_corrections(cleaned)

    # 3. Validate format (informational – does not discard non-matching results).
    if _VN_PLATE_PATTERN.match(corrected):
        return corrected  # well-formed plate

    # Return best-effort result even when format doesn't match perfectly.
    return corrected


# ---------------------------------------------------------------------------
# Step 4 – End-to-end entry point
# ---------------------------------------------------------------------------

def read_license_plate(image_path: str) -> str:
    """
    End-to-end pipeline: load a cropped plate image → preprocess → OCR →
    normalise → return the plate string.

    Args:
        image_path: Path to the cropped license-plate image file.

    Returns:
        Detected plate string (e.g. "30A12345"), or an empty string if
        recognition failed.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    import os
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")

    preprocessed = preprocess_plate(image)
    raw_text = run_ocr(preprocessed)
    plate = normalize_plate_text(raw_text)

    return plate


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ocr.plate_ocr <path_to_plate_image>")
        sys.exit(1)

    path = sys.argv[1]
    result = read_license_plate(path)
    if result:
        print(f"Detected plate: {result}")
    else:
        print("Could not read plate text.")
