from pathlib import Path
import json
from collections import defaultdict

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

COCO_PATH = BASE_DIR / "data" / "annotations" / "annotations_coco.json"
OCR_DIR = BASE_DIR / "data" / "ocr"
IMAGE_DIR = BASE_DIR / "data" / "authentic"
OUTPUT_PATH = BASE_DIR / "data" / "layoutlm_token_dataset.jsonl"

# Only use these classes for Branch 1
TARGET_CLASSES = {
    "gpa": "GPA",
    "issue_date": "ISSUE_DATE",
    "grade_table": "GRADE_TABLE",
    "logo": "LOGO",
    "registrar_block": "REGISTRAR_BLOCK",
}

LABELS = [
    "O",
    "B-GPA", "I-GPA",
    "B-ISSUE_DATE", "I-ISSUE_DATE",
    "B-GRADE_TABLE", "I-GRADE_TABLE",
    "B-LOGO", "I-LOGO",
    "B-REGISTRAR_BLOCK", "I-REGISTRAR_BLOCK",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# -----------------------------
# HELPERS
# -----------------------------
def normalize_bbox_to_1000(x1, y1, x2, y2, width, height):
    return [
        int(1000 * x1 / width),
        int(1000 * y1 / height),
        int(1000 * x2 / width),
        int(1000 * y2 / height),
    ]


def coco_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def intersection_over_word(word_box, region_box):
    wx1, wy1, wx2, wy2 = word_box
    rx1, ry1, rx2, ry2 = region_box

    inter_x1 = max(wx1, rx1)
    inter_y1 = max(wy1, ry1)
    inter_x2 = min(wx2, rx2)
    inter_y2 = min(wy2, ry2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    word_area = max(0, wx2 - wx1) * max(0, wy2 - wy1)
    if word_area == 0:
        return 0.0

    return inter_area / word_area


def find_matching_image_path(file_name):
    # COCO file_name may include Label Studio path fragments
    base_name = Path(file_name).name

    # Search recursively so images stored in nested upload folders are found
    for p in IMAGE_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS and p.name == base_name:
            return p

    # If exact filename doesn't match, try matching by suffix.
    # Label Studio filenames often have a UUID prefix like '<uuid>-<page>-<num>.png'
    parts = base_name.split("-")
    if len(parts) >= 2:
        # try last two segments (e.g. '1-1.png')
        suffix = "-".join(parts[-2:])
        for p in IMAGE_DIR.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS and p.name.endswith(suffix):
                return p

    # fallback: try matching by the final token (e.g. '1.png')
    if len(parts) >= 1:
        suffix = parts[-1]
        for p in IMAGE_DIR.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS and p.name.endswith(suffix):
                return p

    return None


def load_ocr_words(ocr_json_path):
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = []
    bboxes = []
    # support two OCR formats: {'image_size': {'width':.., 'height':..}} or top-level 'width'/'height'
    if "image_size" in data:
        img_w = data["image_size"]["width"]
        img_h = data["image_size"]["height"]
    else:
        img_w = data.get("width")
        img_h = data.get("height")

    for item in data["ocr_results"]:
        text = str(item["text"]).strip()
        if not text:
            continue

        bbox = item["bbox"]
        x1 = float(bbox["x_min"])
        y1 = float(bbox["y_min"])
        x2 = float(bbox["x_max"])
        y2 = float(bbox["y_max"])

        if x2 <= x1 or y2 <= y1:
            continue

        words.append(text)
        bboxes.append([x1, y1, x2, y2])

    return img_w, img_h, words, bboxes


def assign_bio_labels(word_boxes, regions, overlap_threshold=0.5):
    """
    regions = list of dicts:
    {
      "label_name": "GPA",
      "bbox": [x1, y1, x2, y2]
    }
    """
    ner_tags = ["O"] * len(word_boxes)

    # sort regions top-to-bottom then left-to-right for stable BIO assignment
    regions = sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    for region in regions:
        label_name = region["label_name"]
        region_box = region["bbox"]

        matched_indices = []
        for idx, word_box in enumerate(word_boxes):
            overlap = intersection_over_word(word_box, region_box)
            if overlap >= overlap_threshold:
                matched_indices.append(idx)

        if not matched_indices:
            continue

        matched_indices = sorted(matched_indices)
        ner_tags[matched_indices[0]] = f"B-{label_name}"
        for idx in matched_indices[1:]:
            ner_tags[idx] = f"I-{label_name}"

    return ner_tags


# -----------------------------
# MAIN
# -----------------------------
def main():
    if not COCO_PATH.exists():
        raise FileNotFoundError(f"Missing COCO file: {COCO_PATH}")

    if not OCR_DIR.exists():
        raise FileNotFoundError(f"Missing OCR dir: {OCR_DIR}")

    with open(COCO_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = coco["categories"]
    images = coco["images"]
    annotations = coco["annotations"]

    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}
    image_id_to_info = {img["id"]: img for img in images}

    anns_by_image_id = defaultdict(list)
    for ann in annotations:
        image_id = ann["image_id"]
        cat_name = cat_id_to_name[ann["category_id"]]

        if cat_name not in TARGET_CLASSES:
            continue

        anns_by_image_id[image_id].append({
            "label_name": TARGET_CLASSES[cat_name],
            "bbox": coco_xywh_to_xyxy(ann["bbox"]),
        })

    num_written = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for image_id, img_info in image_id_to_info.items():
            image_path = find_matching_image_path(img_info["file_name"])
            if image_path is None:
                print(f"Skipping image not found locally: {img_info['file_name']}")
                continue

            ocr_json_path = OCR_DIR / f"{image_path.stem}.json"
            if not ocr_json_path.exists():
                print(f"Skipping OCR not found: {ocr_json_path.name}")
                continue

            width, height, words, raw_word_boxes = load_ocr_words(ocr_json_path)
            if not words:
                print(f"Skipping empty OCR: {ocr_json_path.name}")
                continue

            regions = anns_by_image_id.get(image_id, [])
            ner_tags_str = assign_bio_labels(raw_word_boxes, regions, overlap_threshold=0.5)

            norm_boxes = [
                normalize_bbox_to_1000(x1, y1, x2, y2, width, height)
                for x1, y1, x2, y2 in raw_word_boxes
            ]
            ner_tags = [LABEL2ID[tag] for tag in ner_tags_str]

            record = {
                "id": str(image_id),
                "image_path": str(image_path),
                "words": words,
                "bboxes": norm_boxes,
                "ner_tags": ner_tags,
                "ner_tags_str": ner_tags_str,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

    print(f"Saved {num_written} records to {OUTPUT_PATH}")
    print("Label mapping:")
    print(json.dumps(LABEL2ID, indent=2))


if __name__ == "__main__":
    main()