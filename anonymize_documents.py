from pathlib import Path
import cv2
import json
import re
import fitz

BASE_DIR = Path(__file__).resolve().parents[1]
AUTHENTIC_DIR = BASE_DIR / "data" / "authentic"
OCR_DIR = BASE_DIR / "data" / "ocr"

PERSONAL_FIELDS = [
    "First Name",
    "Last Name",
]

PLACEHOLDER = "////"


def find_text(ocr_data, keyword):
    results = []
    for item in ocr_data:
        if keyword.lower() in item["text"].lower():
            results.append(item)
    return results


def erase_and_replace(img, bbox, placeholder):
    x1, y1 = int(bbox["x_min"]), int(bbox["y_min"])
    x2, y2 = int(bbox["x_max"]), int(bbox["y_max"])
    
    if x2 <= x1 or y2 <= y1:
        return False
        
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
    
    text_y = y1 + int((y2 - y1) * 0.7)
    font_scale = max(0.4, (y2 - y1) / 100)
    
    cv2.putText(img, placeholder, (x1 + 5, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), 1, cv2.LINE_AA)
    
    return True


def get_name_bbox_for_field(ocr_results, label_keyword):
    for item in ocr_results:
        text = item.get("text", "")
        
        if label_keyword.lower() not in text.lower() or ":" not in text:
            continue
        
        bbox = item.get("bbox", {})
        parts = text.split(":", 1)
        
        x_min = bbox.get("x_min", 0)
        y_min = bbox.get("y_min", 0)
        x_max = bbox.get("x_max", 0)
        y_max = bbox.get("y_max", 0)
        
        width = x_max - x_min
        
        if len(parts) > 1 and parts[1].strip():
            colon_pos = text.find(":")
            label_ratio = (colon_pos + 1) / len(text)
            new_x = int(x_min + label_ratio * width) + 5
            return {
                "x_min": new_x,
                "y_min": y_min,
                "x_max": min(x_max + 50, new_x + 150),
                "y_max": y_max
            }
        else:
            return {
                "x_min": x_max + 5,
                "y_min": y_min,
                "x_max": x_max + 150,
                "y_max": y_max
            }
    
    return None


def process_image(img_path, ocr_path, output_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error reading {img_path}")
        return False
    
    with open(ocr_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    ocr_results = data.get("ocr_results", [])
    
    modified = False
    
    for field in PERSONAL_FIELDS:
        name_bbox = get_name_bbox_for_field(ocr_results, field)
        
        if name_bbox:
            if erase_and_replace(img, name_bbox, PLACEHOLDER):
                modified = True
    
    if modified:
        cv2.imwrite(str(output_path), img)
        print(f"Anonymized: {img_path.name} -> {output_path.name}")
        return True
    
    return False


PDF_PATTERNS = [
    (r"(First Name:)\s*\S+", f"\g<1> {PLACEHOLDER}"),
    (r"(Last Name:)\s*\S+", f"\g<1> {PLACEHOLDER}"),
]


def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    
    for page in doc:
        text = page.get_text()
        
        for pattern, replacement in PDF_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        match = re.search(r"First Name:\s*" + re.escape(PLACEHOLDER), text)
        if match:
            found_text = match.group(0)
            rect = page.search_for(found_text)
            if rect:
                page.add_redact_annot(rect[0], fill=(1, 1, 1))
                page.apply_redactions()
        
        match = re.search(r"Last Name:\s*" + re.escape(PLACEHOLDER), text)
        if match:
            found_text = match.group(0)
            rect = page.search_for(found_text)
            if rect:
                page.add_redact_annot(rect[0], fill=(1, 1, 1))
                page.apply_redactions()
    
    temp_path = pdf_path.with_suffix('.temp.pdf')
    doc.save(temp_path)
    doc.close()
    
    temp_path.replace(pdf_path)
    
    print(f"Anonymized PDF: {pdf_path.name}")
    return True


def main():
    all_png = list(AUTHENTIC_DIR.glob("*.png"))
    image_files = [f for f in all_png if "-" in f.stem and f.stem.count("-") == 1]
    
    processed = 0
    skipped = 0
    
    for img_path in image_files:
        stem = img_path.stem
        ocr_path = OCR_DIR / f"{stem}.json"
        
        if not ocr_path.exists():
            print(f"Skipping {img_path.name} - no OCR file")
            skipped += 1
            continue
        
        if "_fake_" in stem:
            print(f"Skipping {img_path.name} - already synthetic")
            skipped += 1
            continue
        
        output_path = img_path
        
        if process_image(img_path, ocr_path, output_path):
            processed += 1
        else:
            skipped += 1
    
    pdf_files = list(AUTHENTIC_DIR.glob("*.pdf"))
    for pdf_path in pdf_files:
        if "_fake_" in pdf_path.stem:
            continue
        try:
            process_pdf(pdf_path)
            processed += 1
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            skipped += 1
    
    print(f"\nDONE: Processed {processed}, Skipped {skipped}")


if __name__ == "__main__":
    main()