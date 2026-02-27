# License Plate Recognition – OCR Module

OCR pipeline for Vietnamese license plates.  
Accepts a **pre-cropped** plate image and returns the recognised plate string
(e.g. `30A12345`, `51H67890`).

> YOLO detection is **not** included; drop your cropped plate images into
> `resources/` and feed them to `read_license_plate()`.

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `opencv-python-headless` is used instead of `opencv-python` to
> avoid display-library conflicts in server environments.  Replace with
> `opencv-python` if you need GUI windows.

---

## File structure

```
.
├── ocr/
│   ├── __init__.py          # package exports
│   └── plate_ocr.py         # full OCR pipeline
├── resources/               # put your cropped plate images here
├── tests/
│   └── test_plate_ocr.py    # unit tests (pytest)
├── requirements.txt
└── README.md
```

---

## Quick start

```python
from ocr.plate_ocr import read_license_plate

result = read_license_plate("resources/plate.jpg")
print(result)   # e.g. "30A12345"
```

Or from the command line:

```bash
python -m ocr.plate_ocr resources/plate.jpg
```

---

## API

### `preprocess_plate(image: np.ndarray) -> np.ndarray`

Applies the preprocessing pipeline to a NumPy image array:

1. Grayscale conversion
2. Resize (upscale to ≥ 64 px height)
3. Contrast enhancement (CLAHE)
4. Noise removal (median blur)
5. Binarisation (Otsu's threshold)
6. Deskew (rotation correction)

### `run_ocr(image: np.ndarray) -> str`

Runs EasyOCR on a preprocessed image and returns the raw concatenated text.

**Why EasyOCR?**
- Single `pip install`, no external binaries.
- Decent alphanumeric accuracy out of the box.
- Easy to switch to GPU (`gpu=True`) or add more language models.

### `normalize_plate_text(text: str) -> str`

Cleans and standardises raw OCR output:

- Strips non-alphanumeric characters.
- Converts to uppercase.
- Applies **positional** letter↔digit corrections (O→0, I→1, B→8, S→5, …)
  based on the expected VN plate structure: `[2 digits][1-2 letters][4-5 digits]`.
- Validates against Vietnamese plate format.

### `read_license_plate(image_path: str) -> str`

End-to-end helper: loads image → preprocesses → OCR → normalises → returns
plate string.

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

Tests that call EasyOCR are automatically skipped when the library is not
installed, so the preprocessing and normalisation tests always run.

---

## Extending the pipeline

| Goal | How |
|---|---|
| Improve accuracy on noisy images | Swap `_denoise` to `cv2.fastNlMeansDenoising` |
| Handle uneven lighting | Replace Otsu with adaptive threshold in `_threshold` |
| Use GPU | Pass `gpu=True` to `easyocr.Reader` in `_get_reader` |
| Try PaddleOCR | Replace `run_ocr` body with PaddleOCR inference call |
| Add Vietnamese characters | Add `"vi"` to the EasyOCR language list |
