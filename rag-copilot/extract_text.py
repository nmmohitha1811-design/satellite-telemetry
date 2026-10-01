import os
from pypdf import PdfReader

os.makedirs("extracted_text", exist_ok=True)

for filename in os.listdir("documents"):
    if not filename.endswith(".pdf"):
        continue

    pdf_path = os.path.join("documents", filename)
    reader = PdfReader(pdf_path)

    full_text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        full_text += page_text + "\n"

    txt_filename = filename.replace(".pdf", ".txt")
    txt_path = os.path.join("extracted_text", txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    word_count = len(full_text.split())
    page_count = len(reader.pages)
    print(f"{filename}: {page_count} pages, {word_count} words -> {txt_filename}")

print("Done.")