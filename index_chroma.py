from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import sys

# ✅ Automatically detect correct file path
if len(sys.argv) > 1:
    DOCS_FILE = Path(sys.argv[1])
else:
    DOCS_FILE = Path("university_docs.json")
    if not DOCS_FILE.exists():
        DOCS_FILE = Path("/mnt/data/university_docs.json")

assert DOCS_FILE.exists(), f"Docs file not found: {DOCS_FILE}"
print("Using docs file:", DOCS_FILE)

print("Loading SentenceTransformer model (this will download ~100MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Starting Chroma client and collection...")
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
collection = client.get_or_create_collection(name="university_kb")

docs = json.loads(DOCS_FILE.read_text(encoding="utf-8"))

BATCH = 64
ids, texts, metadatas, embeddings = [], [], [], []

for i in tqdm(range(0, len(docs), BATCH), desc="Prepare batches"):
    batch = docs[i:i+BATCH]
    batch_texts = [d["text"] for d in batch]
    batch_ids = [d["id"] for d in batch]
    batch_md = [d["metadata"] for d in batch]
    embs = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
    # add to chorma in one go
    collection.add(documents=batch_texts, metadatas=batch_md, ids=batch_ids, embeddings=embs.tolist())

print("Persisting Chroma DB to ./chroma_db")
client.persist()
print("Done. Indexed", len(docs), "documents.")
