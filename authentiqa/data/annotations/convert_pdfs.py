"""Convert PDFs to PNG images for Label Studio."""
import os
import fitz

AUTHENTIC_DIR = os.path.join(os.path.dirname(__file__), "..", "authentic")

for i in range(1, 21):
    pdf_path = os.path.join(AUTHENTIC_DIR, f"{i}.pdf")
    if os.path.exists(pdf_path):
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)
            output_path = os.path.join(AUTHENTIC_DIR, f"{i}-{page_num + 1}.png")
            pix.save(output_path)
            print(f"Saved: {output_path}")
        doc.close()

print("PDF conversion complete!")
