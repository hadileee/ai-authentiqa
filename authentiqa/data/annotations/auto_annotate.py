"""Programmatic annotation using PDF analysis and pattern matching."""
import os
import json
import fitz
import numpy as np
import cv2
from PIL import Image
import io

AUTHENTIC_DIR = os.path.join(os.path.dirname(__file__), "..", "authentic")
ANNOTATIONS_DIR = os.path.dirname(__file__)

CATEGORIES = [
    {"id": 1, "name": "logo"},
    {"id": 2, "name": "signature"},
    {"id": 3, "name": "stamp"},
    {"id": 4, "name": "registrar_block"},
    {"id": 5, "name": "gpa"},
    {"id": 6, "name": "issue_date"},
    {"id": 7, "name": "grade_table"},
]

def get_page_text_blocks(page):
    """Get text blocks with positions from PDF page."""
    blocks = page.get_text("dict")["blocks"]
    text_blocks = []
    for block in blocks:
        if "lines" in block:
            x0 = block["bbox"][0]
            y0 = block["bbox"][1]
            x1 = block["bbox"][2]
            y1 = block["bbox"][3]
            text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    text += span["text"] + " "
            text_blocks.append({
                "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
                "text": text.strip().lower(),
                "bbox": (x0, y0, x1, y1)
            })
    return text_blocks

def classify_region(block, page_height):
    """Classify a text block based on keywords."""
    text = block["text"].lower()
    
    y_normalized = block["y"] / page_height
    
    if any(kw in text for kw in ["gpa", "grade point", "cgpa", "sgpa"]):
        return "gpa"
    if any(kw in text for kw in ["date", "issued", "issue date", "dated"]):
        return "issue_date"
    if any(kw in text for kw in ["registrar", "controller", "examination", "dean"]):
        return "registrar_block"
    if any(kw in text for kw in ["course", "credit", "grade", "semester"]) and len(text) > 10:
        return "grade_table"
    
    if y_normalized < 0.15:
        return "logo"
    elif y_normalized > 0.85:
        return "signature"
    elif y_normalized > 0.7 and y_normalized < 0.85:
        return "stamp"
    
    return None

def detect_regions_from_image(img_array, page_height):
    """Detect regions using image analysis."""
    height, width = img_array.shape[:2]
    regions = []
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / (width * height)
        
        if 0.01 < area_ratio < 0.3:
            y_normalized = y / height
            
            if y_normalized < 0.2:
                label = "logo"
            elif y_normalized > 0.8:
                label = "signature"
            elif y_normalized > 0.7:
                label = "stamp"
            else:
                continue
            
            regions.append({
                "label": label,
                "bbox": [float(x), float(y), float(w), float(h)]
            })
    
    return regions

def annotate_transcript(pdf_path, doc_id):
    """Annotate a single transcript."""
    doc = fitz.open(pdf_path)
    annotations = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        page_width = page.rect.width
        
        text_blocks = get_page_text_blocks(page)
        
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        img_array = np.array(img)
        
        image_regions = detect_regions_from_image(img_array, page_height)
        
        classified_blocks = []
        for block in text_blocks:
            label = classify_region(block, page_height)
            if label:
                x_scale = img_array.shape[1] / page_width
                y_scale = img_array.shape[0] / page_height
                
                x = int(block["x"] * x_scale)
                y = int(block["y"] * y_scale)
                w = int(block["width"] * x_scale)
                h = int(block["height"] * y_scale)
                
                classified_blocks.append({
                    "id": len(classified_blocks) + 1,
                    "image_id": page_num + 1,
                    "category_id": next(c["id"] for c in CATEGORIES if c["name"] == label),
                    "label": label,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "text_preview": block["text"][:50] if block["text"] else ""
                })
        
        for region in image_regions:
            if not any(b["label"] == region["label"] for b in classified_blocks):
                classified_blocks.append({
                    "id": len(classified_blocks) + 1,
                    "image_id": page_num + 1,
                    "category_id": next(c["id"] for c in CATEGORIES if c["name"] == region["label"]),
                    "label": region["label"],
                    "bbox": [int(x) for x in region["bbox"]],
                    "area": int(region["bbox"][2] * region["bbox"][3]),
                    "text_preview": "(image detected)"
                })
        
        annotations.extend(classified_blocks)
    
    doc.close()
    return annotations

def create_coco_format(annotations_dict):
    """Convert to COCO format."""
    images = []
    annotations = []
    ann_id = 1
    
    for img_name, img_anns in annotations_dict.items():
        base_name = os.path.basename(img_name)
        img_id = int(base_name.split(".")[0])
        images.append({
            "id": img_id,
            "file_name": img_name,
            "width": 1275,
            "height": 1650
        })
        
        for ann in img_anns:
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": 0,
                "label": ann["label"],
                "text": ann.get("text_preview", "")
            })
            ann_id += 1
    
    return {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }

def main():
    print("Starting programmatic annotation...")
    
    all_annotations = {}
    
    for i in range(1, 21):
        pdf_path = os.path.join(AUTHENTIC_DIR, f"{i}.pdf")
        if os.path.exists(pdf_path):
            print(f"Annotating {i}.pdf...")
            annotations = annotate_transcript(pdf_path, i)
            all_annotations[f"{i}.png"] = annotations
            print(f"  Found {len(annotations)} regions")
    
    coco_format = create_coco_format(all_annotations)
    
    output_path = os.path.join(ANNOTATIONS_DIR, "annotations_coco.json")
    with open(output_path, "w") as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"\nSaved COCO annotations to: {output_path}")
    print(f"Total images: {len(coco_format['images'])}")
    print(f"Total annotations: {len(coco_format['annotations'])}")
    
    print("\nWARNING: This is AUTO-ANNOTATION using pattern matching.")
    print("You MUST manually verify and correct bounding boxes in Label Studio.")
    print("\nLabels found:")
    for cat in CATEGORIES:
        count = sum(1 for a in coco_format["annotations"] if a["label"] == cat["name"])
        print(f"  {cat['name']}: {count}")

if __name__ == "__main__":
    main()
