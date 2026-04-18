"""
Authentiqa Bounding Box Annotation Tool
A simple GUI tool to annotate transcript regions.
"""
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import fitz

AUTHENTIC_DIR = os.path.join(os.path.dirname(__file__), "..", "authentic")
ANNOTATIONS_DIR = os.path.dirname(__file__)

LABELS = ["logo", "signature", "stamp", "registrar_block", "gpa", "issue_date", "grade_table"]
LABEL_COLORS = {
    "logo": "#FF6B6B",
    "signature": "#4ECDC4",
    "stamp": "#FFE66D",
    "registrar_block": "#95E1D3",
    "gpa": "#A8E6CF",
    "issue_date": "#DDA0DD",
    "grade_table": "#87CEEB"
}

class AnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Authentiqa Annotation Tool")
        self.root.geometry("1400x900")
        
        self.current_doc = 1
        self.current_page = 1
        self.max_pages = {}
        self.annotations = {}
        self.current_label = tk.StringVar(value="logo")
        self.rect_start = None
        self.canvas_rects = []
        
        self.load_annotations()
        self.scan_documents()
        self.setup_ui()
        self.load_image()
    
    def load_annotations(self):
        """Load existing annotations."""
        self.annotations_file = os.path.join(ANNOTATIONS_DIR, "manual_annotations.json")
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, "r") as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {}
    
    def save_annotations(self):
        """Save annotations to file."""
        with open(self.annotations_file, "w") as f:
            json.dump(self.annotations, f, indent=2, default=str)
    
    def scan_documents(self):
        """Scan for available documents."""
        self.docs = []
        for i in range(1, 26):
            pdf_path = os.path.join(AUTHENTIC_DIR, f"{i}.pdf")
            if os.path.exists(pdf_path):
                doc = fitz.open(pdf_path)
                self.max_pages[i] = len(doc)
                doc.close()
                self.docs.append(i)
    
    def setup_ui(self):
        """Setup the UI layout."""
        control_frame = ttk.Frame(self.root, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Document:", font=("Arial", 12, "bold")).pack(pady=5)
        
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(pady=5)
        
        ttk.Button(nav_frame, text="◀", width=3, command=self.prev_doc).pack(side=tk.LEFT)
        self.doc_label = ttk.Label(nav_frame, text=f"{self.current_doc}/20", font=("Arial", 14, "bold"))
        self.doc_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="▶", width=3, command=self.next_doc).pack(side=tk.LEFT)
        
        ttk.Label(control_frame, text="Page:", font=("Arial", 12, "bold")).pack(pady=5)
        
        page_frame = ttk.Frame(control_frame)
        page_frame.pack(pady=5)
        
        ttk.Button(page_frame, text="◀", width=3, command=self.prev_page).pack(side=tk.LEFT)
        self.page_label = ttk.Label(page_frame, text="1/1", font=("Arial", 12))
        self.page_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(page_frame, text="▶", width=3, command=self.next_page).pack(side=tk.LEFT)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=15)
        
        ttk.Label(control_frame, text="Label:", font=("Arial", 12, "bold")).pack(pady=5)
        
        for label in LABELS:
            color = LABEL_COLORS[label]
            rb = ttk.Radiobutton(
                control_frame, text=label, variable=self.current_label, value=label,
                style="Label.Radiobutton"
            )
            rb.pack(anchor='w', padx=20)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=15)
        
        ttk.Button(control_frame, text="Delete Selected", command=self.delete_selected).pack(pady=5)
        ttk.Button(control_frame, text="Clear All", command=self.clear_all).pack(pady=5)
        ttk.Button(control_frame, text="Export COCO", command=self.export_coco).pack(pady=5)
        
        self.status_label = ttk.Label(control_frame, text="", foreground="blue")
        self.status_label.pack(pady=10)
        
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="#404040", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        scroll_y = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Delete>", self.delete_selected)
        self.canvas.bind("<Next>", self.next_page)
        self.canvas.bind("<Prior>", self.prev_page)
    
    def load_image(self):
        """Load current document page as image."""
        pdf_path = os.path.join(AUTHENTIC_DIR, f"{self.current_doc}.pdf")
        
        if not os.path.exists(pdf_path):
            return
        
        doc = fitz.open(pdf_path)
        page = doc[self.current_page - 1]
        pix = page.get_pixmap(dpi=150)
        doc.close()
        
        img_data = pix.tobytes("png")
        self.current_image = Image.open(io.BytesIO(img_data))
        self.current_image_path = os.path.join(AUTHENTIC_DIR, f"{self.current_doc}-{self.current_page}.png")
        self.current_image.save(self.current_image_path)
        
        self.photo = ImageTk.PhotoImage(self.current_image)
        
        self.canvas.delete("all")
        self.canvas.configure(width=self.photo.width(), height=self.photo.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        self.draw_annotations()
        
        self.doc_label.config(text=f"{self.current_doc}/20")
        self.page_label.config(text=f"{self.current_page}/{self.max_pages.get(self.current_doc, 1)}")
        
        self.update_status()
    
    def draw_annotations(self):
        """Draw all annotations for current page."""
        self.canvas_rects = []
        
        key = f"{self.current_doc}-{self.current_page}"
        if key not in self.annotations:
            return
        
        for ann in self.annotations[key]:
            color = LABEL_COLORS.get(ann["label"], "#FFFFFF")
            x, y, w, h = ann["bbox"]
            rect = self.canvas.create_rectangle(x, y, x+w, y+h, outline=color, width=2)
            self.canvas.create_text(x, y-5, text=ann["label"], fill=color, anchor=tk.SW, font=("Arial", 8))
            self.canvas_rects.append({"id": rect, "ann": ann})
    
    def on_click(self, event):
        """Handle mouse click."""
        self.rect_start = (event.x, event.y)
    
    def on_drag(self, event):
        """Handle mouse drag."""
        if self.rect_start:
            self.canvas.delete("temp_rect")
            self.canvas.create_rectangle(
                self.rect_start[0], self.rect_start[1], event.x, event.y,
                outline=LABEL_COLORS[self.current_label.get()], width=2, tags="temp_rect"
            )
    
    def on_release(self, event):
        """Handle mouse release - create annotation."""
        if self.rect_start:
            x1, y1 = self.rect_start
            x2, y2 = event.x, event.y
            
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            
            if w > 10 and h > 10:
                key = f"{self.current_doc}-{self.current_page}"
                if key not in self.annotations:
                    self.annotations[key] = []
                
                ann = {
                    "id": len(self.annotations[key]) + 1,
                    "label": self.current_label.get(),
                    "bbox": [x, y, w, h]
                }
                self.annotations[key].append(ann)
                self.save_annotations()
                self.draw_annotations()
                self.update_status()
            
            self.rect_start = None
            self.canvas.delete("temp_rect")
    
    def delete_selected(self(self):
        """Delete selected annotation."""
        pass
    
    def clear_all(self):
        """Clear all annotations for current page."""
        key = f"{self.current_doc}-{self.current_page}"
        if key in self.annotations and messagebox.askyesno("Confirm", "Clear all annotations on this page?"):
            self.annotations[key] = []
            self.save_annotations()
            self.draw_annotations()
            self.update_status()
    
    def export_coco(self):
        """Export annotations in COCO format."""
        images = []
        annotations = []
        ann_id = 1
        
        for doc_id in self.docs:
            for page_num in range(1, self.max_pages.get(doc_id, 1) + 1):
                key = f"{doc_id}-{page_num}"
                
                images.append({
                    "id": (doc_id - 1) * 5 + page_num,
                    "file_name": f"{doc_id}-{page_num}.png",
                    "width": self.current_image.width if self.current_doc == doc_id and self.current_page == page_num else 1275,
                    "height": self.current_image.height if self.current_doc == doc_id and self.current_page == page_num else 1650
                })
                
                if key in self.annotations:
                    for ann in self.annotations[key]:
                        x, y, w, h = ann["bbox"]
                        annotations.append({
                            "id": ann_id,
                            "image_id": (doc_id - 1) * 5 + page_num,
                            "category_id": LABELS.index(ann["label"]) + 1,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                            "label": ann["label"]
                        })
                        ann_id += 1
        
        categories = [{"id": i+1, "name": name} for i, name in enumerate(LABELS)]
        
        coco = {"images": images, "annotations": annotations, "categories": categories}
        
        output_path = os.path.join(ANNOTATIONS_DIR, "annotations_coco.json")
        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)
        
        messagebox.showinfo("Export", f"Exported {len(annotations)} annotations to:\n{output_path}")
    
    def update_status(self):
        """Update status bar."""
        key = f"{self.current_doc}-{self.current_page}"
        count = len(self.annotations.get(key, []))
        self.status_label.config(text=f"Annotations: {count}")
    
    def prev_doc(self):
        """Go to previous document."""
        idx = self.docs.index(self.current_doc) if self.current_doc in self.docs else 0
        if idx > 0:
            self.current_doc = self.docs[idx - 1]
            self.current_page = 1
            self.load_image()
    
    def next_doc(self):
        """Go to next document."""
        idx = self.docs.index(self.current_doc) if self.current_doc in self.docs else 0
        if idx < len(self.docs) - 1:
            self.current_doc = self.docs[idx + 1]
            self.current_page = 1
            self.load_image()
    
    def prev_page(self, event=None):
        """Go to previous page."""
        if self.current_page > 1:
            self.current_page -= 1
            self.load_image()
    
    def next_page(self, event=None):
        """Go to next page."""
        max_p = self.max_pages.get(self.current_doc, 1)
        if self.current_page < max_p:
            self.current_page += 1
            self.load_image()

if __name__ == "__main__":
    import io
    root = tk.Tk()
    app = AnnotationApp(root)
    root.mainloop()
