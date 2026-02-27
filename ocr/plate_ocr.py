"""
OCR module for Vietnamese license plate recognition using YOLOv5 character detection.

Based on: https://github.com/trungdinh22/License-Plate-Recognition

The OCR stage uses a YOLOv5 model specifically trained to detect individual
characters on Vietnamese license plates, then assembles them positionally.
This approach is more accurate than general-purpose OCR libraries for this domain.

Pipeline:
  1. deskew_plate(image)              – rotation correction using HoughLinesP
  2. run_ocr(image)                   – YOLOv5 character detection & plate assembly
  3. normalize_plate_text(text)       – clean and standardise the raw OCR output
  4. read_license_plate(image_path)   – end-to-end helper (tries 4 deskew variants)

  preprocess_plate(image) is also provided for optional grayscale preprocessing.

Model:
  Download LP_ocr.pt (full) or LP_ocr_nano_62.pt (faster/lighter) from Google Drive:
  https://drive.google.com/file/d/... (see README for the exact link)
  Place the file at model/LP_ocr.pt, or pass model_path to read_license_plate().

Installation:
  pip install -r requirements.txt

Usage example:
  from ocr.plate_ocr import read_license_plate
  result = read_license_plate("resources/plate.jpg")
  print(result)   # e.g. "30A12345"
"""

import math
import re
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Default model path
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH = "model/LP_ocr.pt"

# Lazy-loaded YOLOv5 character-detection model.
_ocr_model = None


def _get_ocr_model(model_path: str = _DEFAULT_MODEL_PATH):
    """
    Return a cached YOLOv5 character-detection model loaded from model_path.

    torch is imported lazily so the module can be imported without PyTorch
    installed (e.g. in environments that only run preprocessing tests).
    """
    global _ocr_model
    if _ocr_model is None:
        import torch  # noqa: PLC0415
        _ocr_model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=model_path,
            force_reload=False,
            verbose=False,
        )
        _ocr_model.conf = 0.60
    return _ocr_model


# ---------------------------------------------------------------------------
# Deskew utilities
# Ported from utils_rotate.py in trungdinh22/License-Plate-Recognition
# ---------------------------------------------------------------------------

def _change_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE in LAB colour space to enhance contrast on a BGR image."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle degrees around its centre."""
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


def _compute_skew(src_img: np.ndarray, center_threshold: int) -> float:
    """
    Estimate the skew angle of src_img using HoughLinesP on Canny edges.

    center_threshold=1 skips lines whose centre y-coordinate is less than 7 px
    (useful for 2-line plates where a line near the top acts as a separator).
    Returns 0.0 if no suitable lines are detected.
    """
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    else:
        h, w = src_img.shape

    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img, threshold1=30, threshold2=100,
                      apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(
        edges, 1, math.pi / 180, 30,
        minLineLength=w / 1.5,
        maxLineGap=h / 3.0,
    )
    if lines is None:
        return 0.0

    # Find the line whose centre is closest to the top of the image.
    min_line_y = 100
    min_line_idx = 0
    for i, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            cy = (y1 + y2) / 2
            if center_threshold == 1 and cy < 7:
                continue
            if cy < min_line_y:
                min_line_y = cy
                min_line_idx = i

    angle_sum = 0.0
    count = 0
    for x1, y1, x2, y2 in lines[min_line_idx]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30:
            angle_sum += ang
            count += 1

    if count == 0:
        return 0.0
    return (angle_sum / count) * 180 / math.pi


def deskew_plate(src_img: np.ndarray,
                 change_contrast: bool = False,
                 center_threshold: int = 0) -> np.ndarray:
    """
    Correct skew in a plate image using Hough line detection.

    Args:
        src_img: BGR plate image (NumPy array).
        change_contrast: If True, apply CLAHE before skew estimation.
        center_threshold: Passed to _compute_skew; set to 1 to ignore lines
                          near the top (useful for 2-row plates).

    Returns:
        Rotated BGR image with skew corrected.
    """
    if change_contrast:
        return _rotate_image(src_img, _compute_skew(_change_contrast(src_img), center_threshold))
    return _rotate_image(src_img, _compute_skew(src_img, center_threshold))


# ---------------------------------------------------------------------------
# Step 1 – Optional preprocessing pipeline (grayscale path)
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
    the aspect ratio.
    """
    h, w = image.shape[:2]
    if h >= target_height:
        return image
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)


def _enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to a grayscale image to boost local contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def _denoise(image: np.ndarray) -> np.ndarray:
    """Remove noise with a fast median blur."""
    return cv2.medianBlur(image, 3)


def _threshold(image: np.ndarray) -> np.ndarray:
    """Binarise the image with Otsu's thresholding."""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _deskew(image: np.ndarray) -> np.ndarray:
    """
    Correct small rotations in a binary grayscale image using minAreaRect.
    Skips correction when the tilt is negligible (<0.5°).
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
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_plate(image: np.ndarray) -> np.ndarray:
    """
    Optional grayscale preprocessing pipeline for a cropped plate image.

    Steps applied in order:
      1. Grayscale conversion
      2. Resize (upscale to at least 128 px height)
      3. Contrast enhancement via CLAHE
      4. Noise removal via median blur
      5. Binarisation via Otsu's threshold
      6. Deskew (rotation correction via minAreaRect)

    Args:
        image: NumPy array – BGR, BGRA, or grayscale plate image.

    Returns:
        Preprocessed grayscale binary image (NumPy array, dtype uint8).

    Note:
        The main OCR pipeline (run_ocr / read_license_plate) works on the
        original BGR image.  This function is provided for inspection or
        use with non-YOLOv5 backends.
    """
    img = _to_grayscale(image)
    img = _resize(img, target_height=128)
    img = _enhance_contrast(img)
    img = _denoise(img)
    img = _threshold(img)
    img = _deskew(img)
    return img


# ---------------------------------------------------------------------------
# Character assembly helper
# Ported from helper.py in trungdinh22/License-Plate-Recognition
# ---------------------------------------------------------------------------

def _linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b


def _check_point_linear(x, y, x1, y1, x2, y2):
    a, b = _linear_equation(x1, y1, x2, y2)
    y_pred = a * x + b
    return math.isclose(y_pred, y, abs_tol=3)


def _assemble_plate_from_detections(bb_list) -> str:
    """
    Assemble a plate string from YOLOv5 character bounding-box detections.

    Each entry in bb_list is [xmin, ymin, xmax, ymax, conf, class_id, class_name].
    Handles both 1-line and 2-line (stacked) Vietnamese plates.

    Returns the assembled plate string, or "unknown" if the detection count is
    outside the expected range (7–10 characters).
    """
    if len(bb_list) == 0 or not (7 <= len(bb_list) <= 10):
        return "unknown"

    center_list = []
    y_sum = 0
    for bb in bb_list:
        x_c = (bb[0] + bb[2]) / 2
        y_c = (bb[1] + bb[3]) / 2
        y_sum += y_c
        center_list.append([x_c, y_c, bb[-1]])  # [cx, cy, char_label]

    # Find leftmost and rightmost character centres.
    l_point = min(center_list, key=lambda c: c[0])
    r_point = max(center_list, key=lambda c: c[0])

    # Determine if 2-line plate: any character not on the line through l and r.
    lp_type = "1"
    if l_point[0] != r_point[0]:
        for ct in center_list:
            if not _check_point_linear(
                ct[0], ct[1],
                l_point[0], l_point[1],
                r_point[0], r_point[1],
            ):
                lp_type = "2"
                break

    y_mean = int(y_sum / len(bb_list))
    license_plate = ""
    if lp_type == "2":
        line_1 = [c for c in center_list if int(c[1]) <= y_mean]
        line_2 = [c for c in center_list if int(c[1]) > y_mean]
        for ch in sorted(line_1, key=lambda c: c[0]):
            license_plate += str(ch[2])
        license_plate += "-"
        for ch in sorted(line_2, key=lambda c: c[0]):
            license_plate += str(ch[2])
    else:
        for ch in sorted(center_list, key=lambda c: c[0]):
            license_plate += str(ch[2])

    return license_plate


# ---------------------------------------------------------------------------
# Step 2 – OCR using YOLOv5 character detection
# ---------------------------------------------------------------------------

def run_ocr(image: np.ndarray, model_path: str = _DEFAULT_MODEL_PATH) -> str:
    """
    Run YOLOv5 character detection on a plate image and return the assembled text.

    Uses a YOLOv5 model specifically trained on Vietnamese license plate characters
    (from https://github.com/trungdinh22/License-Plate-Recognition).

    Args:
        image: BGR NumPy array of the (optionally deskewed) plate crop.
        model_path: Path to the LP_ocr.pt YOLOv5 weights file.

    Returns:
        Assembled plate string (e.g. "30A12345") or "unknown" if fewer than 7
        or more than 10 characters are detected.
    """
    model = _get_ocr_model(model_path)
    results = model(image)
    bb_list = results.pandas().xyxy[0].values.tolist()
    return _assemble_plate_from_detections(bb_list)


# ---------------------------------------------------------------------------
# Step 3 – Post-processing / normalisation
# ---------------------------------------------------------------------------

# Vietnamese license plate format patterns:
#   Standard (1-letter series):           [2 digits][1 letter][4-5 digits]        e.g. 30A12345
#   One-row (2-letter series):            [2 digits][2 letters][4-5 digits]       e.g. 51AB1234
#   Newer (letter+digit series, e.g. G1): [2 digits][1 letter][1 digit][5 digits] e.g. 29G133333
_VN_PLATE_PATTERN = re.compile(
    r"^(\d{2})([A-Z]\d|[A-Z]{1,2})(\d{4,5})$"
)


def _apply_positional_corrections(plate: str) -> str:
    """
    Apply letter↔digit substitutions based on position in the plate string.

    Vietnamese plates follow: <2-digit province code> <1-2 letter series> <4-5 digit number>
    Newer plates also use:    <2-digit province code> <1 letter + 1 digit series> <5 digit number>
    e.g. 29G133333 where 'G1' is the series code.

    - Province code (positions 0-1): must be digits → convert look-alike letters.
    - Series (positions 2-3 or 2): must be letters → convert look-alike digits.
      Exception: in a letter+digit series the character at position 3 is a digit and
      is left unchanged.
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
    # After the 2-digit province code the series is:
    #   - 1-2 letters (classic format), or
    #   - 1 letter followed by 1 digit (newer letter+digit series, total plate length 9).
    series_end = 2
    if length > 2 and chars[2].isalpha():
        series_end = 3
        if length > 3 and chars[3].isalpha():
            series_end = 4  # 2-letter series (e.g. AB)
        elif length > 3 and chars[3].isdigit() and length == 9:
            series_end = 4  # letter+digit series (e.g. G1): position 3 is the series digit

    # Series characters must be letters; the digit part of a letter+digit series stays as-is.
    for i in range(2, series_end):
        is_series_digit = (series_end == 4 and i == 3 and chars[i].isdigit())
        if not is_series_digit:
            chars[i] = letter_fixes.get(chars[i], chars[i])

    # Sequence (after the series) must be digits.
    for i in range(series_end, length):
        chars[i] = digit_fixes.get(chars[i], chars[i])

    return "".join(chars)


def _try_recover_valid_plate(plate: str) -> str:
    """
    Attempt to recover a valid plate string from one that is too long (likely
    caused by an extra detection due to noise or a separator artefact).
    Tries removing each character in turn and returns the first candidate that,
    after positional corrections, matches the Vietnamese plate pattern.
    Returns an empty string if no valid candidate is found.
    """
    for i in range(len(plate)):
        candidate = plate[:i] + plate[i + 1:]
        candidate = _apply_positional_corrections(candidate)
        if _VN_PLATE_PATTERN.match(candidate):
            return candidate
    return ""


def normalize_plate_text(text: str) -> str:
    """
    Clean and standardise raw OCR output into a Vietnamese license plate string.

    Steps:
      1. Uppercase and strip whitespace / non-alphanumeric characters.
      2. Apply positional letter↔digit corrections.
      3. Validate against the expected VN plate format (optional – returns best
         guess even if it does not match, so the caller can decide).
      4. If the string is too long (> 9 chars), attempt recovery by removing one
         character at a time until a valid format is found.

    Args:
        text: Raw OCR string (may contain spaces, punctuation, noise, or a
              dash separator for 2-line plates e.g. "30A-12345").

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

    # 4. If too long (> 9 chars), try removing one character to recover a valid plate.
    #    This handles cases where an extra character is detected from noise or a
    #    plate-separator artefact (e.g. a dash or dot read as an extra character).
    if len(corrected) > 9:
        recovered = _try_recover_valid_plate(corrected)
        if recovered:
            return recovered

    # Return best-effort result even when format doesn't match perfectly.
    return corrected


# ---------------------------------------------------------------------------
# Step 4 – End-to-end entry point
# ---------------------------------------------------------------------------

def read_license_plate(image_path: str,
                       model_path: str = _DEFAULT_MODEL_PATH) -> str:
    """
    End-to-end pipeline: load a cropped plate image → try up to 4 deskew
    variants → YOLOv5 character detection → normalise → return plate string.

    The 4 deskew variants cover combinations of contrast enhancement and
    centre-threshold filtering, ensuring robust handling of skewed or
    unevenly-lit plates (1-line and 2-line formats).

    Args:
        image_path: Path to the cropped license-plate image file.
        model_path: Path to the LP_ocr.pt YOLOv5 weights file
                    (default: "model/LP_ocr.pt").

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

    # Try up to 4 deskew combinations (change_contrast × center_threshold).
    for cc in range(2):
        for ct in range(2):
            deskewed = deskew_plate(image, change_contrast=bool(cc), center_threshold=ct)
            raw_text = run_ocr(deskewed, model_path=model_path)
            if raw_text != "unknown":
                plate = normalize_plate_text(raw_text)
                if plate:
                    return plate

    return ""


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ocr.plate_ocr <path_to_plate_image> [model_path]")
        sys.exit(1)

    path = sys.argv[1]
    mpath = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_MODEL_PATH
    result = read_license_plate(path, model_path=mpath)
    if result:
        print(f"Detected plate: {result}")
    else:
        print("Could not read plate text.")
