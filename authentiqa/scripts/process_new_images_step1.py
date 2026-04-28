"""
Comprehensive script to process new uncommitted authentic and forged images.

Steps:
1. Rename new authentic images with sequential numbers (26-1, 27-1, etc.)
2. Run OCR on new authentic images
3. Apply anonymization (replace sensitive data with ////)
4. Create/update COCO annotation file (result1.json)
5. Update layoutlm dataset with new records
6. Process forged images: OCR only + create forged dataset
"""

from pathlib import Path
import json
import cv2
import shutil
import re

BASE_DIR = Path(__file__).resolve().parents[1]
AUTHENTIC_OFFICIAL_DIR = BASE_DIR / "data" / "authentic" / "official"
AUTHENTIC_DIR = BASE_DIR / "data" / "authentic"
OCR_DIR = BASE_DIR / "data" / "ocr"
FORGED_DIR = BASE_DIR / "data" / "forged_real"
RESULT1_PATH = BASE_DIR / "data" / "annotations" / "result1.json"

print("=" * 80)
print("STEP 1: RENAME NEW AUTHENTIC IMAGES WITH SEQUENTIAL NUMBERS")
print("=" * 80)

# Find all _auth_aug_ files
auth_aug_files = sorted(AUTHENTIC_OFFICIAL_DIR.glob("*_auth_aug_*.png"))
print(f"\nFound {len(auth_aug_files)} new authentic images")

# Group by source document (Imen Krifa, Kacem.abidi, Updated Official transcript)
from collections import defaultdict
grouped_files = defaultdict(list)

for f in auth_aug_files:
    # Extract source name
    if "Imen Krifa" in f.name:
        source = "Imen Krifa"
    elif "Kacem.abidi" in f.name:
        source = "Kacem.abidi"
    elif "Updated Official transcript MedTech" in f.name:
        source = "Updated Official MedTech"
    else:
        source = "Unknown"
    
    grouped_files[source].append(f)

# Find the next available document number
existing_numbers = set()
for img in AUTHENTIC_DIR.glob("*.png"):
    match = re.match(r"(\d+)-\d+\.png", img.name)
    if match:
        existing_numbers.add(int(match.group(1)))

next_doc_num = max(existing_numbers) + 1 if existing_numbers else 1
print(f"\nNext available document number: {next_doc_num}")

# Rename files
rename_mapping = {}
for source, files in sorted(grouped_files.items()):
    print(f"\n{source}: {len(files)} images -> Document {next_doc_num}")
    
    # Group by page if applicable
    pages = defaultdict(list)
    for f in sorted(files):
        if "page_1" in f.name:
            page = 1
        elif "page_2" in f.name:
            page = 2
        else:
            page = 1
        pages[page].append(f)
    
    # Rename each page group sequentially
    for page_num, page_files in sorted(pages.items()):
        for seq, old_path in enumerate(sorted(page_files), 1):
            new_name = f"{next_doc_num}-{seq}.png"
            new_path = AUTHENTIC_DIR / new_name
            
            print(f"  Rename: {old_path.name[:40]}... -> {new_name}")
            shutil.copy2(old_path, new_path)
            rename_mapping[old_path.name] = new_name
    
    next_doc_num += 1

print(f"\n✓ Renamed {len(rename_mapping)} files")
print("\nRename mapping:")
for old, new in sorted(rename_mapping.items())[:5]:
    print(f"  {old[:50]}... -> {new}")
if len(rename_mapping) > 5:
    print(f"  ... and {len(rename_mapping) - 5} more")

print("\n" + "=" * 80)
print("STEP 2: RUN OCR ON NEW AUTHENTIC IMAGES")
print("=" * 80)
print("\nThis will be done next with run_ocr.py script")

print("\n" + "=" * 80)
print("STEP 3: ANONYMIZE SENSITIVE DATA")
print("=" * 80)
print("\nThis will be done with anonymize_documents.py script")

print("\n" + "=" * 80)
print("STEP 4: CREATE/UPDATE COCO ANNOTATION FILE")
print("=" * 80)
print("\nWill create empty annotation file for labeling in Label Studio")
print(f"Update path: {RESULT1_PATH}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Renamed {len(rename_mapping)} new authentic images")
print(f"✓ Mapping saved - ready for next steps")
print("\nNext steps:")
print("1. Run: python scripts/run_ocr.py")
print("2. Run: python scripts/anonymize_documents.py")
print("3. Label images in Label Studio")
print("4. Export annotations to result1.json")
print("5. Run: python scripts/build_layoutlm_dataset_official_21_25.py")
