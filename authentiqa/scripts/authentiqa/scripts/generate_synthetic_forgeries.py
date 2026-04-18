from pathlib import Path
import json
import random
import cv2

# ===== PATHS =====
BASE_DIR = Path(__file__).resolve().parents[1]

IMG_DIR = BASE_DIR / "data" / "authentic"
OCR_DIR = BASE_DIR / "data" / "ocr"

OUT_IMG_DIR = BASE_DIR / "data" / "synthetic" / "images"
OUT_META_DIR = BASE_DIR / "data" / "synthetic" / "metadata"

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_META_DIR.mkdir(parents=True, exist_ok=True)

# ===== HELPERS =====

def find_text(ocr_data, keyword):
    results = []
    for item in ocr_data:
        if keyword.lower() in item["text"].lower():
            results.append(item)
    return results


def find_numbers_near(ocr_data, ref_bbox):
    cx = (ref_bbox["x_min"] + ref_bbox["x_max"]) / 2
    cy = (ref_bbox["y_min"] + ref_bbox["y_max"]) / 2

    candidates = []

    for item in ocr_data:
        text = item["text"]
        if any(c.isdigit() for c in text):
            b = item["bbox"]
            x = (b["x_min"] + b["x_max"]) / 2
            y = (b["y_min"] + b["y_max"]) / 2

            dist = abs(x - cx) + abs(y - cy)
            candidates.append((dist, item))

    candidates.sort(key=lambda x: x[0])
    return [c[1] for c in candidates[:3]]


def erase_region(img, bbox):
    x1, y1 = int(bbox["x_min"]), int(bbox["y_min"])
    x2, y2 = int(bbox["x_max"]), int(bbox["y_max"])
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)


def write_text(img, text, bbox):
    x1, y1 = int(bbox["x_min"]), int(bbox["y_min"])
    cv2.putText(img, text, (x1, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 2, cv2.LINE_AA)


# ===== FORGERY OPS =====

def alter_gpa(img, ocr_data):
    gpa_items = find_text(ocr_data, "GPA")
    if not gpa_items:
        return None

    gpa_box = gpa_items[0]["bbox"]
    numbers = find_numbers_near(ocr_data, gpa_box)

    if not numbers:
        return None

    target = numbers[0]
    old = target["text"]

    try:
        val = float(old.replace(",", "."))
        new_val = round(val + random.uniform(0.5, 3.0), 2)
        new_text = str(new_val)
    except:
        return None

    erase_region(img, target["bbox"])
    write_text(img, new_text, target["bbox"])

    return {
        "type": "gpa_alteration",
        "old": old,
        "new": new_text,
        "bbox": target["bbox"]
    }


def alter_date(img, ocr_data):
    date_items = find_text(ocr_data, "date")

    if not date_items:
        return None

    target = date_items[0]
    old = target["text"]

    new_text = f"{random.randint(1,28)}/{random.randint(1,12)}/20{random.randint(20,25)}"

    erase_region(img, target["bbox"])
    write_text(img, new_text, target["bbox"])

    return {
        "type": "date_alteration",
        "old": old,
        "new": new_text,
        "bbox": target["bbox"]
    }


# ===== MAIN =====

def process_file(img_path, ocr_path, num_versions=10):
    img = cv2.imread(str(img_path))

    with open(ocr_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ocr_data = data["ocr_results"]

    for i in range(num_versions):
        img_copy = img.copy()
        ops_applied = []

        ops = [alter_gpa, alter_date]
        random.shuffle(ops)

        for op in ops[:random.randint(1, 2)]:
            result = op(img_copy, ocr_data)
            if result:
                ops_applied.append(result)

        if not ops_applied:
            continue

        out_name = f"{img_path.stem}_fake_{i}.png"

        out_img_path = OUT_IMG_DIR / out_name
        out_meta_path = OUT_META_DIR / f"{img_path.stem}_fake_{i}.json"

        cv2.imwrite(str(out_img_path), img_copy)

        meta = {
            "source": img_path.name,
            "output": out_name,
            "label": "forged",
            "operations": ops_applied
        }

        with open(out_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def main():
    images = list(IMG_DIR.glob("*.*"))

    for img_path in images:
        ocr_path = OCR_DIR / f"{img_path.stem}.json"
        if not ocr_path.exists():
            continue

        process_file(img_path, ocr_path, num_versions=10)

    print("DONE: Synthetic data generated")


if __name__ == "__main__":
    main()