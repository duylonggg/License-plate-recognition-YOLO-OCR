# License Plate Recognition – YOLOv5 OCR Module

OCR pipeline for Vietnamese license plates using a **YOLOv5 model trained specifically on Vietnamese plate characters**.  
Based on the approach from [trungdinh22/License-Plate-Recognition](https://github.com/trungdinh22/License-Plate-Recognition) which is more accurate than general-purpose OCR libraries for this domain.

Accepts a **pre-cropped** plate image and returns the recognised plate string  
(e.g. `30A12345`, `51AB1234`, `29G133333`).

> YOLO detection is **not** included in this module; drop your cropped plate images into  
> `resources/` and feed them to `read_license_plate()`.

---

## How it works

```
Input image → deskew (×4 attempts) → YOLOv5 character detection → assemble → normalise → plate string
```

1. **Deskew** – Hough line detection corrects rotation (tries 4 combinations of contrast-enhancement and centre-threshold filtering).
2. **Character detection** – A YOLOv5 model detects individual characters as bounding boxes.
3. **Assembly** – Characters are sorted by X-coordinate; 2-line plates are detected via Y-coordinate analysis.
4. **Normalise** – Positional letter↔digit corrections (`O→0`, `I→1`, `S→5`, …) are applied based on the expected Vietnamese plate structure.

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `opencv-python-headless` is used instead of `opencv-python` to avoid  
> display-library conflicts in server/CI environments. Replace with `opencv-python`  
> if you need GUI windows (e.g. for `cv2.imshow`).

---

## Model setup

This module requires a pretrained YOLOv5 OCR model. Two variants are available from  
[trungdinh22/License-Plate-Recognition](https://github.com/trungdinh22/License-Plate-Recognition):

| File | Size | Speed |
|------|------|-------|
| `LP_ocr.pt` | Larger | Higher accuracy |
| `LP_ocr_nano_62.pt` | Smaller | Faster (~15–20 fps on CPU with 1 plate) |

**Steps:**

1. Download `LP_ocr.pt` (or `LP_ocr_nano_62.pt`) from the  
   [Google Drive link in the reference repo](https://github.com/trungdinh22/License-Plate-Recognition).
2. Create a `model/` directory in the project root and place the file there:

```
model/
└── LP_ocr.pt
```

3. *(Optional)* Pass a custom path via the `model_path` argument:

```python
result = read_license_plate("resources/plate.jpg", model_path="model/LP_ocr_nano_62.pt")
```

> On first run, `torch.hub` will automatically download YOLOv5 weights/code  
> (cached in `~/.cache/torch/hub`). Subsequent runs use the cache.

---

## File structure

```
.
├── model/
│   └── LP_ocr.pt            # pretrained YOLOv5 OCR model (download separately)
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
# With a custom model path:
python -m ocr.plate_ocr resources/plate.jpg model/LP_ocr_nano_62.pt
```

---

## API

### `read_license_plate(image_path, model_path="model/LP_ocr.pt") → str`

End-to-end helper: loads image → tries 4 deskew variants → YOLOv5 character
detection → normalises → returns plate string.

- Raises `FileNotFoundError` if the file does not exist.
- Raises `ValueError` if the file cannot be decoded as an image.
- Returns `""` if no valid plate is detected.

### `deskew_plate(image, change_contrast=False, center_threshold=0) → np.ndarray`

Corrects skew in a BGR plate image using Hough line detection.  
`change_contrast=True` applies CLAHE before skew estimation (helps on dark or  
low-contrast plates). `center_threshold=1` ignores lines near the top edge  
(useful for 2-line plates).

### `run_ocr(image, model_path="model/LP_ocr.pt") → str`

Runs the YOLOv5 character-detection model on a (deskewed) plate image and  
returns the assembled plate string or `"unknown"` if fewer than 7 or more than  
10 characters are detected.

### `normalize_plate_text(text) → str`

Cleans and standardises raw OCR output:

- Strips non-alphanumeric characters; uppercases.
- Applies **positional** letter↔digit corrections based on the expected VN plate  
  structure: `[2 digits][1–2 letter series][4–5 digits]`.
- Handles newer `[letter+digit series]` plates (e.g. `29G133333`).
- If the string is > 9 characters, attempts recovery by removing one character.

### `preprocess_plate(image) → np.ndarray`

Optional grayscale preprocessing pipeline (grayscale → resize → CLAHE →  
median blur → Otsu threshold → deskew). Useful for visualisation or  
non-YOLOv5 backends.

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

Tests covering preprocessing, deskew utilities, character assembly, and  
normalisation run without PyTorch or the model file. The end-to-end  
integration test is automatically skipped when `torch` is not installed  
or `model/LP_ocr.pt` is absent.

---

## Performance tips

| Goal | How |
|------|-----|
| Faster inference | Use `LP_ocr_nano_62.pt` instead of `LP_ocr.pt` |
| GPU acceleration | Install CUDA-enabled PyTorch; model runs on GPU automatically |
| Batch processing | Call `run_ocr` directly and reuse the cached model (`_get_ocr_model()`) |
| Noisy / dark plates | Pass `change_contrast=True` to `deskew_plate` |
| 2-line plates | The pipeline handles them automatically via Y-coordinate analysis |

