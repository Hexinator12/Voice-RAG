# app_rag.py
import json
import os
from pathlib import Path
from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from dotenv import load_dotenv

load_dotenv()  # read .env for OPENAI_API_KEY, if present
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set. Set it in your environment or .env file to use OpenAI.")
else:
    openai.api_key = OPENAI_API_KEY

# Load FAISS index and metadata
INDEX_PATH = "university.faiss.index"
META_PATH = "university_meta.json"
assert Path(INDEX_PATH).exists(), f"Missing index file: {INDEX_PATH}"
assert Path(META_PATH).exists(), f"Missing metadata file: {META_PATH}"

print("Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
ids = meta["ids"]
texts = meta["texts"]
metadatas = meta["metadatas"]

# Sentence transformer
print("Loading encoder model...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI(title="University Voice-RAG Backend")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    max_tokens: int = 256
    temperature: float = 0.0

class RetrievedChunk(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    source_chunks: List[RetrievedChunk]

def retrieve(query: str, top_k: int = 5):
    q_emb = encoder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
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

def build_prompt(query: str, retrieved: List[dict]):
    # Keep it short and deterministic
    context = ""
    for i, r in enumerate(retrieved, start=1):
        context += f"[{i}] (id:{r['id']}) {r['text']}\n\n"
    system = (
        "You are an assistant that answers user questions about the university using ONLY the provided context. "
        "If the answer is not contained in the context, say 'I don't know'. Be concise and give sources by id in square brackets."
    )
    user = f"User question: {query}\n\nContext:\n{context}\n\nRespond concisely and add source ids."
    return system, user

async def call_openai_chat(system_prompt: str, user_prompt: str, max_tokens: int = 256, temperature: float = 0.0):
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
    # using chat completions (gpt-4/4o or gpt-4o-mini etc). change model as needed.
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # change to gpt-4o, gpt-4, or text-davinci if you prefer; pick one you have access to
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"].strip()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not req.query or req.query.strip() == "":
        raise HTTPException(status_code=400, detail="Empty query")
    retrieved = retrieve(req.query, top_k=req.top_k)
    if len(retrieved) == 0:
        return {"answer": "I don't know.", "source_chunks": []}
    system_prompt, user_prompt = build_prompt(req.query, retrieved)
    try:
        answer = await call_openai_chat(system_prompt, user_prompt, max_tokens=req.max_tokens, temperature=req.temperature)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")
    # return answer and retrieved chunks (for citation/source)
    return {
        "answer": answer,
        "source_chunks": retrieved
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_rag:app", host="127.0.0.1", port=8001, reload=True)
