import json
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading embedding model (first run downloads it, ~80MB)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

print(f"Embedding {len(texts)} chunks...")
embeddings = model.encode(texts, show_progress_bar=True)

print("Embedding shape:", embeddings.shape)

np.save("chunk_embeddings.npy", embeddings)

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print("Saved chunk_embeddings.npy")   