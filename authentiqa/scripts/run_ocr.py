from pathlib import Path
import json
import easyocr
import cv2
from tqdm import tqdm

# === PATHS ===
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "data" / "authentic"
OUTPUT_DIR = BASE_DIR / "data" / "ocr"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === INIT OCR ===
reader = easyocr.Reader(['en'], gpu=False)

VALID_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]


def run_ocr(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read {image_path}")
        return None

    h, w = img.shape[:2]

    results = reader.readtext(img)

    ocr_data = []

    for (bbox, text, conf) in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]

        ocr_data.append({
            "text": text,
            "confidence": float(conf),
            "bbox": {
                "x_min": int(min(xs)),
                "y_min": int(min(ys)),
                "x_max": int(max(xs)),
                "y_max": int(max(ys))
            }
        })

    return {
        "file_name": image_path.name,
        "width": w,
        "height": h,
        "ocr_results": ocr_data
    }


def main():
    images = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in VALID_EXTS]

    if not images:
        print("No images found in data/authentic/")
        return

    for img_path in tqdm(images, desc="Running OCR"):
        result = run_ocr(img_path)

        if result is None:
            continue

        out_file = OUTPUT_DIR / f"{img_path.stem}.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print("DONE: OCR results saved in data/ocr/")


if __name__ == "__main__":
    main()