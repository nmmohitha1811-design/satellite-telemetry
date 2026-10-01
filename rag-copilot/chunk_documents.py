import os
import json

CHUNK_SIZE = 200      # target words per chunk
CHUNK_OVERLAP = 40    # words repeated between consecutive chunks

def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap
    return chunks

all_chunks = []

for filename in os.listdir("extracted_text"):
    if not filename.endswith(".txt"):
        continue

    path = os.path.join("extracted_text", filename)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    doc_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    for i, chunk in enumerate(doc_chunks):
        all_chunks.append({
            "id": f"{filename}::chunk{i}",
            "source": filename,
            "chunk_index": i,
            "text": chunk
        })

    print(f"{filename}: {len(doc_chunks)} chunks")

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

print(f"Total chunks: {len(all_chunks)}")
print("Saved to chunks.json")