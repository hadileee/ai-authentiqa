#!/usr/bin/env python3
import os
from pathlib import Path
import fitz 
import cv2
import json
import easyocr
from tqdm import tqdm
import re

# === PATHS ===
BASE_DIR = Path(__file__).resolve().parents[1]
OFFICIAL_PDF = BASE_DIR / "data" / "authentic" / "official" / "21.pdf"
AUTHENTIC_DIR = BASE_DIR / "data" / "authentic"
OCR_DIR = BASE_DIR / "data" / "ocr"

AUTHENTIC_DIR.mkdir(parents=True, exist_ok=True)
OCR_DIR.mkdir(parents=True, exist_ok=True)

# === CONFIG ===
PERSONAL_FIELDS = ["First name", "Last name", "CIN", "Student ID"]
PLACEHOLDER = "////"

# === OCR PROCESSING ===
def run_ocr_on_images(image_paths):
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
    x1, y1, x2, y2 = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
    # White out original text
    cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 5, y2 + 2), (255, 255, 255), -1)
    # Add ////
    font_scale = max(0.5, (y2 - y1) / 60)
    cv2.putText(img, PLACEHOLDER, (x1, y1 + int((y2-y1)*0.8)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return True

def process_image_anonymize(img_path, ocr_path):
    img = cv2.imread(str(img_path))
    if img is None: return False
    
    with open(ocr_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    ocr_results = data.get("ocr_results", [])
    modified = False
    
    for field in PERSONAL_FIELDS:
        for item in ocr_results:
            text = item["text"].lower()
            if field.lower() in text:
                # If name is in the same box (e.g. "First name: Kacem")
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

                # If name is in a separate box to the right
                label_bbox = item["bbox"]
                center_y = (label_bbox["y_min"] + label_bbox["y_max"]) / 2
                
                for potential_val in ocr_results:
                    pbox = potential_val["bbox"]
                    # Check if box is to the right and vertically aligned
                    if pbox["x_min"] > label_bbox["x_max"] and abs(((pbox["y_min"]+pbox["y_max"])/2) - center_y) < 20:
                        # This is the value box (e.g. "Kacem")
                        if erase_and_replace(img, pbox):
                            modified = True
                            break
    
    if modified:
        cv2.imwrite(str(img_path), img)
    return modified

# === MAIN ===
def main():
    # 1. Convert PDF
    print("1️⃣ Converting PDF to Images...")
    doc = fitz.open(OFFICIAL_PDF)
    images = []
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=300)
        p = AUTHENTIC_DIR / f"21-{i+1}.png"
        pix.save(str(p))
        images.append(p)
    doc.close()

    # 2. Run OCR (The part where you get stuck)
    print("2️⃣ Running OCR (Please wait, CPU is slow)...")
    run_ocr_on_images(images)
    
    # 3. Anonymize
    print("3️⃣ Replacing names with ////...")
    for img_path in images:
        ocr_p = OCR_DIR / f"{img_path.stem}.json"
        if ocr_p.exists():
            process_image_anonymize(img_path, ocr_p)
            print(f"✅ Processed {img_path.name}")

    print("\n🎉 All done! Check your PNG files.")

if __name__ == "__main__":
    main()