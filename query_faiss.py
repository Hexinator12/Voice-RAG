# query_faiss.py
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("university.faiss.index")
with open("university_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

ids = meta["ids"]
texts = meta["texts"]
metadatas = meta["metadatas"]

def search(query, top_k=5):
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)  # D: scores, I: indices
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append({
            "id": ids[idx],
            "score": float(score),
            "text": texts[idx],
            "metadata": metadatas[idx]
        })
    return results

if __name__ == "__main__":
    while True:
        q = input("Query (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        res = search(q, top_k=5)
        for r in res:
            print(f"\nID: {r['id']}  score:{r['score']:.4f}")
            print(r['text'][:400].replace("\n"," ") + ("..." if len(r['text'])>400 else ""))
