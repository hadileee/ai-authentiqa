"""
Label Studio Project Setup Script
Usage: python setup_label_studio.py <API_TOKEN>
"""
import base64
import os
import requests

AUTHENTIC_DIR = os.path.join(os.path.dirname(__file__), "..", "authentic")
ANNOTATIONS_DIR = os.path.dirname(__file__)

LABEL_STUDIO_URL = "http://localhost:8080"

LABEL_CONFIG = '''<View>
  <Image name="data" value="$data"/>
  
  <RectangleLabels name="label" toName="data">
    <Label value="logo" background="#FF6B6B"/>
    <Label value="signature" background="#4ECDC4"/>
    <Label value="stamp" background="#FFE66D"/>
    <Label value="registrar_block" background="#95E1D3"/>
    <Label value="gpa" background="#A8E6CF"/>
    <Label value="issue_date" background="#DDA0DD"/>
    <Label value="grade_table" background="#87CEEB"/>
  </RectangleLabels>
</View>'''


def create_project(api_token):
    headers = {"Authorization": f"Token {api_token}"}
    payload = {
        "title": "Authentiqa Transcript Annotation",
        "description": "Bounding box annotations for authentic transcript regions"
    }
    response = requests.post(f"{LABEL_STUDIO_URL}/api/projects/", headers=headers, json=payload)
    if response.status_code in [200, 201]:
        project = response.json()
        print(f"[OK] Created project: {project['title']} (ID: {project['id']})")
        return project['id']
    else:
        print(f"[ERROR] Failed to create project: {response.text}")
        return None


def set_labeling_config(project_id, api_token):
    headers = {"Authorization": f"Token {api_token}"}
    payload = {"label_config": LABEL_CONFIG}
    response = requests.post(f"{LABEL_STUDIO_URL}/api/projects/{project_id}/update", headers=headers, json=payload)
    if response.status_code == 200:
        print("[OK] Labeling config updated")
        return True
    else:
        print(f"[ERROR] Failed to update config: {response.text}")
        return False


def import_images(project_id, api_token):
    from label_studio_sdk import Client
    ls_client = Client(url=LABEL_STUDIO_URL, token=api_token)
    
    image_files = sorted([f for f in os.listdir(AUTHENTIC_DIR) if f.endswith('.png')])
    print(f"[INFO] Found {len(image_files)} PNG images")
    
    tasks = []
    for img_file in image_files:
        img_path = os.path.join(AUTHENTIC_DIR, img_file)
        with open(img_path, 'rb') as f:
            b64_data = base64.b64encode(f.read()).decode('utf-8')
            tasks.append({"data": f"data:image/png;base64,{b64_data}"})
    
    ls_client.tasks.create(project=project_id, tasks=tasks)
    
    print(f"[OK] Imported {len(image_files)} images")
    return True


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python setup_label_studio.py <API_TOKEN>")
        return
    
    api_token = sys.argv[1]
    print("=" * 50)
    print("Label Studio Project Setup")
    print("=" * 50)
    
    print("\n[1] Creating project...")
    project_id = create_project(api_token)
    if not project_id:
        return
    
    print("\n[2] Setting labeling config...")
    set_labeling_config(project_id, api_token)
    
    print("\n[3] Importing images...")
    import_images(project_id, api_token)
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("=" * 50)
    print(f"\nOpen: http://localhost:8080/projects/{project_id}/")
    print("\nAnnotate all images, then export as COCO JSON")


if __name__ == "__main__":
    main()