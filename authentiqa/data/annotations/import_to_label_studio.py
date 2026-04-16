"""Import pre-annotations into Label Studio via API."""
import os
import json
import requests

ANNOTATIONS_FILE = os.path.join(os.path.dirname(__file__), "annotations_coco.json")
LABEL_STUDIO_URL = "http://localhost:8080"

def import_annotations(api_token, project_id):
    """Import COCO annotations into Label Studio project."""
    headers = {"Authorization": f"Token {api_token}"}
    
    with open(ANNOTATIONS_FILE, "r") as f:
        coco_data = json.load(f)
    
    for image in coco_data["images"]:
        img_id = image["id"]
        file_name = image["file_name"]
        
        task_payload = {
            "project": project_id,
            "data": {"transcript": f"/data/upload/{file_name}"}
        }
        
        response = requests.post(
            f"{LABEL_STUDIO_URL}/api/tasks/",
            headers=headers,
            json=task_payload
        )
        
        if response.status_code in [200, 201]:
            task = response.json()
            task_id = task["id"]
            
            img_annotations = [a for a in coco_data["annotations"] if a["image_id"] == img_id]
            
            ls_annotations = []
            for ann in img_annotations:
                category = next(c for c in coco_data["categories"] if c["id"] == ann["category_id"])
                x, y, w, h = ann["bbox"]
                
                ls_annotations.append({
                    "id": len(ls_annotations) + 1,
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "transcript",
                    "original_width": image["width"],
                    "original_height": image["height"],
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": (x / image["width"]) * 100,
                        "y": (y / image["height"]) * 100,
                        "width": (w / image["width"]) * 100,
                        "height": (h / image["height"]) * 100,
                        "rectanglelabels": [category["name"]]
                    },
                    "score": 0.9
                })
            
            if ls_annotations:
                annotation_payload = {
                    "task": task_id,
                    "completed_by": 1,
                    "annotations": [{"result": ls_annotations}]
                }
                
                requests.post(
                    f"{LABEL_STUDIO_URL}/api/annotations/",
                    headers=headers,
                    json=annotation_payload
                )
            
            print(f"Imported: {file_name} ({len(img_annotations)} annotations)")
        else:
            print(f"Failed to create task for {file_name}: {response.text}")

def main():
    print("Import Pre-annotations to Label Studio")
    print("=" * 50)
    
    api_token = input("Enter your Label Studio API token: ").strip()
    project_id = input("Enter your project ID: ").strip()
    
    if not api_token or not project_id:
        print("API token and project ID are required.")
        return
    
    import_annotations(api_token, int(project_id))
    print("\nImport complete!")

if __name__ == "__main__":
    main()
