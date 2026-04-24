#!/usr/bin/env python3
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import re

# === PATHS ===
BASE_DIR = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = BASE_DIR / "data" / "authentic" / "official"
AUTHENTIC_DIR = BASE_DIR / "data" / "authentic"
OCR_DIR = BASE_DIR / "data" / "ocr"

AUTHENTIC_DIR.mkdir(parents=True, exist_ok=True)
OCR_DIR.mkdir(parents=True, exist_ok=True)

# === CONFIG ===
PERSONAL_FIELDS = [
    "First name",
    "Last name",
    "CIN",
    "Student ID",
    # Common OCR variants seen in 24/25
    "name",        # e.g. "name:"
    "first",       # e.g. "First njme"
    "last",        # e.g. "Lst name"
]
PLACEHOLDER = "////"

# === OCR PROCESSING ===
def run_ocr_on_images(image_paths):
    import cv2
    import easyocr

    # Initialize reader once
    reader = easyocr.Reader(['en'], gpu=False)
    
    for img_path in tqdm(image_paths, desc="🔍 Running OCR"):
        img = cv2.imread(str(img_path))
        if img is None: continue

        results = reader.readtext(str(img_path))
        ocr_data = []
        for (bbox, text, conf) in results:
            xs, ys = [p[0] for p in bbox], [p[1] for p in bbox]
            ocr_data.append({
                "text": text,
                "bbox": {"x_min": int(min(xs)), "y_min": int(min(ys)), 
                         "x_max": int(max(xs)), "y_max": int(max(ys))}
            })

        out_file = OCR_DIR / f"{img_path.stem}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"ocr_results": ocr_data}, f, indent=2)
    
    # Clean up reader memory
    del reader

# === ANONYMIZATION ===
def erase_and_replace(img, bbox):
    import cv2

    x1, y1, x2, y2 = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
    # White out original text
    cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 5, y2 + 2), (255, 255, 255), -1)
    # Add ////
    font_scale = max(0.5, (y2 - y1) / 60)
    cv2.putText(img, PLACEHOLDER, (x1, y1 + int((y2-y1)*0.8)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return True

def process_image_anonymize(img_path, ocr_path):
    import cv2

    def is_personal_field_label(label_text: str) -> bool:
        t = label_text.lower().strip()
        # Broad matching for noisy OCR variants
        if "cin" in t or re.fullmatch(r"cin[:\s]*", t):
            return True
        if "student" in t and ("id" in t or "ld" in t):
            return True
        if "first" in t and ("name" in t or "njme" in t or "nme" in t):
            return True
        if ("last" in t or "lst" in t) and "name" in t:
            return True
        # Signature/name lines at the bottom (e.g. Programs Manager → person name)
        if "program" in t and "manager" in t:
            return True
        if "director" in t and "deputy" in t:
            return True
        if re.search(r"\bname\b", t) and ":" in t:
            return True
        # Fallback: legacy simple contains check
        return any(f.lower() in t for f in PERSONAL_FIELDS)

    img = cv2.imread(str(img_path))
    if img is None: return False
    
    with open(ocr_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    ocr_results = data.get("ocr_results", [])
    modified = False
    
    for item in ocr_results:
        if not is_personal_field_label(item.get("text", "")):
            continue

        # If value is in the same box (e.g. "First name: Kacem")
        if ":" in item["text"] and len(item["text"].split(":")) > 1:
            val = item["text"].split(":")[1].strip()
            if val:
                width = item["bbox"]["x_max"] - item["bbox"]["x_min"]
                ratio = (item["text"].find(":") + 1) / len(item["text"])
                target_bbox = item["bbox"].copy()
                target_bbox["x_min"] += int(width * ratio)
                if erase_and_replace(img, target_bbox):
                    modified = True
                    continue

        # Otherwise try to find the value box to the right
        label_bbox = item["bbox"]
        center_y = (label_bbox["y_min"] + label_bbox["y_max"]) / 2

        for potential_val in ocr_results:
            pbox = potential_val["bbox"]
            if pbox["x_min"] > label_bbox["x_max"] and abs(((pbox["y_min"] + pbox["y_max"]) / 2) - center_y) < 20:
                if erase_and_replace(img, pbox):
                    modified = True
                    break
    
    if modified:
        cv2.imwrite(str(img_path), img)
    return modified

def process_official_pdf(pdf_path: Path) -> list[Path]:
    import fitz

    doc_id = pdf_path.stem  # e.g. "21"

    # 1. Convert PDF → Images
    print(f"[1/3] Converting {pdf_path.name} to images...")
    doc = fitz.open(pdf_path)
    images: list[Path] = []
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=300)
        p = AUTHENTIC_DIR / f"{doc_id}-{i+1}.png"
        pix.save(str(p))
        images.append(p)
    doc.close()

    # 2. OCR → JSON
    print("[2/3] Running OCR (Please wait, CPU is slow)...")
    run_ocr_on_images(images)

    # 3. Anonymize PNGs (overwrite)
    print("[3/3] Replacing names with ////...")
    for img_path in images:
        ocr_p = OCR_DIR / f"{img_path.stem}.json"
        if ocr_p.exists():
            process_image_anonymize(img_path, ocr_p)
            print(f"Processed {img_path.name}")

    print(f"\nDone: {doc_id}\n")
    return images


# === MAIN ===
def main():
    parser = argparse.ArgumentParser(description="Convert official transcript PDFs to PNGs + OCR JSON + anonymized PNGs.")
    parser.add_argument(
        "--ids",
        nargs="*",
        help="Document ids to process (e.g. 21 22 23). Defaults to all PDFs found in data/authentic/official/.",
    )
    args = parser.parse_args()

    if args.ids:
        pdfs = [OFFICIAL_DIR / f"{doc_id}.pdf" for doc_id in args.ids]
    else:
        pdfs = sorted(OFFICIAL_DIR.glob("*.pdf"), key=lambda p: p.stem)

    missing = [p for p in pdfs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing PDFs: {', '.join(str(p) for p in missing)}")

    for pdf_path in pdfs:
        process_official_pdf(pdf_path)

if __name__ == "__main__":
    main()