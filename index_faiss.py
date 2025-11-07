# index_faiss.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

DOCS_FILE = Path("university_docs.json")   # adjust path if needed
assert DOCS_FILE.exists(), f"Docs file not found: {DOCS_FILE}"

print("Loading docs...")
docs = json.loads(DOCS_FILE.read_text(encoding="utf-8"))
# each doc expected: {'id': ..., 'text': ..., 'metadata': {...}}

texts = [d["text"] for d in docs]
ids = [d["id"] for d in docs]
metadatas = [d.get("metadata", {}) for d in docs]

print("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # same as you used

BATCH = 64
all_embeddings = []

print("Encoding texts in batches...")
for i in tqdm(range(0, len(texts), BATCH), desc="Encode"):
    batch_texts = texts[i:i+BATCH]
    embs = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
    all_embeddings.append(embs)

embeddings = np.vstack(all_embeddings).astype("float32")
# normalize for cosine similarity via inner product
faiss.normalize_L2(embeddings)

dim = embeddings.shape[1]
print(f"Embeddings shape: {embeddings.shape}, dim={dim}")

# Use an inner-product (cosine because vectors normalized) flat index
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print("Total vectors in index:", index.ntotal)

# persist index and metadata
faiss.write_index(index, "university.faiss.index")

meta_out = {
    "ids": ids,
    "metadatas": metadatas,
    "texts": texts
}
with open("university_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta_out, f, ensure_ascii=False, indent=2)

print("Saved FAISS index -> university.faiss.index")
print("Saved metadata -> university_meta.json")
