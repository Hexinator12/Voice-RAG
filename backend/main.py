# backend/main.py
"""
Revised FastAPI backend for Voice-RAG (local vector store).
 - Supports JSON + form for /query
 - Uses new OpenAI SDK (openai>=1.0.0) via OpenAI() client
 - Local vector store (pickle + numpy)
 - Upload / ingest / query / upload-audio endpoints
"""

import os
import uuid
import shutil
import subprocess
import base64
import pickle
import json
import logging
from typing import Optional, List, Dict, Any


from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer

# Optional OpenAI import (new SDK)
try:
    from openai import OpenAI
    _HAS_OPENAI_SDK = True
except Exception:
    _HAS_OPENAI_SDK = False

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("voice-rag")

# --- Directories and paths ---
THIS_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(THIS_DIR, "data")
MODELS_DIR = os.path.join(THIS_DIR, "models")
TMP_DIR = os.path.join(THIS_DIR, "tmp")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# whisper/llama/tts binaries (optional)
WHISPER_CLI = os.path.abspath(os.path.join(ROOT, "whisper.cpp", "build", "bin", "whisper-cli"))
WHISPER_MODEL = os.path.join(MODELS_DIR, "ggml-small.bin")
LLAMA_GGUF_PATH = os.path.abspath(os.path.join(ROOT, "llama.cpp", "models", "mistral-7b-instruct.Q4_0.gguf"))
LLAMA_CLI_BINARY = os.path.abspath(os.path.join(ROOT, "llama.cpp", "build", "bin", "llama-cli"))

# --- Embedding model and local store ---
log.info("Loading SentenceTransformer model (this may download ~100MB if not cached)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
EMB_DIM = embed_model.get_sentence_embedding_dimension()
log.info(f"Embedding dimension: {EMB_DIM}")

STORE_DIR = os.path.join(DATA_DIR, "vector_store")
os.makedirs(STORE_DIR, exist_ok=True)
STORE_META_PATH = os.path.join(STORE_DIR, "meta.pkl")
STORE_EMB_PATH = os.path.join(STORE_DIR, "embeddings.npy")

# in-memory store variables
_store_meta: List[Dict[str, Any]] = []
_store_emb: Optional[np.ndarray] = None

def _load_store():
    global _store_meta, _store_emb
    try:
        if os.path.exists(STORE_META_PATH):
            with open(STORE_META_PATH, "rb") as fh:
                _store_meta = pickle.load(fh)
                log.info(f"Loaded meta: {len(_store_meta)} entries")
        else:
            _store_meta = []
            log.info("No meta.pkl found; starting empty.")
        if os.path.exists(STORE_EMB_PATH):
            _store_emb = np.load(STORE_EMB_PATH)
            log.info(f"Loaded embeddings shape: {_store_emb.shape}")
        else:
            _store_emb = None
            log.info("No embeddings.npy found; embeddings empty.")
    except Exception as e:
        log.exception("Failed loading store; starting fresh.")
        _store_meta = []
        _store_emb = None

def _save_store():
    global _store_meta, _store_emb
    with open(STORE_META_PATH, "wb") as fh:
        pickle.dump(_store_meta, fh)
    if _store_emb is None:
        empty = np.zeros((0, EMB_DIM), dtype=np.float32)
        np.save(STORE_EMB_PATH, empty)
    else:
        np.save(STORE_EMB_PATH, _store_emb)

_load_store()

def add_documents_to_store(documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str], embeddings: List[List[float]]):
    """Persist documents + embeddings to the local store (append)."""
    global _store_meta, _store_emb
    embs = np.array(embeddings, dtype=np.float32)
    if _store_emb is None or _store_emb.size == 0:
        _store_emb = embs.copy()
    else:
        _store_emb = np.vstack([_store_emb, embs])
    for i, doc in enumerate(documents):
        _store_meta.append({"id": ids[i], "document": doc, "metadata": metadatas[i]})
    _save_store()
    log.info(f"Added {len(documents)} docs to store. New meta length: {len(_store_meta)}")

def query_store(query_embeddings: List[List[float]], n_results: int = 4, include: List[str] = ["documents","metadatas","distances"]):
    """Return results in a chroma-ish structure: {'results':[{'ids':[], 'documents':[], 'metadatas':[], 'distances': []}] }"""
    global _store_meta, _store_emb
    q = np.array(query_embeddings, dtype=np.float32)
    results = {"results": []}
    if _store_emb is None or _store_emb.shape[0] == 0:
        for _ in range(q.shape[0]):
            results["results"].append({"ids": [], "documents": [], "metadatas": [], "distances": []})
        return results

    store_emb = _store_emb
    # normalize and compute cosine similarity
    store_norms = np.linalg.norm(store_emb, axis=1, keepdims=True) + 1e-8
    store_normed = store_emb / store_norms
    q_norms = np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    q_normed = q / q_norms
    sims = np.dot(q_normed, store_normed.T)  # similarity [-1,1]
    dists = 1.0 - sims  # lower is better
    for qi in range(q.shape[0]):
        row = dists[qi]
        idx = np.argsort(row)[:n_results]
        ids_out = [ _store_meta[i]["id"] for i in idx ]
        docs_out = [ _store_meta[i]["document"] for i in idx ]
        metas_out = [ _store_meta[i]["metadata"] for i in idx ]
        dists_out = row[idx].tolist()
        results["results"].append({
            "ids": ids_out,
            "documents": docs_out,
            "metadatas": metas_out,
            "distances": dists_out
        })
    return results

# --- Optional Llama python fallback (not required) ---
try:
    from llama_cpp import Llama
    _HAS_LLAMA_PY = True
except Exception:
    _HAS_LLAMA_PY = False

def run_llama_prompt(prompt: str, max_tokens: int = 256):
    """Best-effort Llama call: prefer python binding, fallback to llama-cli binary."""
    if _HAS_LLAMA_PY:
        try:
            llm = Llama(model_path=LLAMA_GGUF_PATH, n_ctx=1024)
            out = llm(prompt, max_tokens=max_tokens)
            return out.get("choices", [{}])[0].get("text", "").strip()
        except Exception as e:
            log.exception("llama_cpp failed")
            return f"LLM (python binding) failed: {e}"
    else:
        if not os.path.exists(LLAMA_CLI_BINARY) or not os.path.exists(LLAMA_GGUF_PATH):
            return "LLM not available: neither llama_cpp binding nor llama-cli/model found."
        cmd = [
            LLAMA_CLI_BINARY,
            "-m", LLAMA_GGUF_PATH,
            "-p", prompt,
            "-n", str(max_tokens),
            "-c", "1024",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            out = proc.stdout.strip() or proc.stderr.strip()
            return out
        except Exception as e:
            log.exception("llama-cli run failed")
            return f"LLM (llama-cli) failed: {e}"

# --- Optional whisper + wav helpers ---
def run_whisper_on_wav(wav_path: str) -> str:
    if not os.path.exists(WHISPER_CLI) or not os.path.exists(WHISPER_MODEL):
        raise FileNotFoundError("whisper-cli or model not found.")
    cmd = [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", wav_path, "--no-prints"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    return out

def wav_from_upload(src_path: str, out_wav: str):
    cmd = ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", out_wav]
    subprocess.run(cmd, check=True)

def tts_say_to_wav(text: str, out_wav: str):
    tmp_aiff = out_wav + ".aiff"
    safe_text = text.replace('"', "'")
    subprocess.run(["say", safe_text, "-o", tmp_aiff], check=True)
    subprocess.run(["ffmpeg", "-y", "-i", tmp_aiff, out_wav], check=True)
    os.remove(tmp_aiff)

# --- OpenAI client wrapper (new SDK) ---
# Load .env (safe)
try:
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    try:
        load_dotenv()
    except Exception:
        pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # default; change in .env

if _HAS_OPENAI_SDK and OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    log.info("OpenAI client initialized.")
else:
    openai_client = None
    if not _HAS_OPENAI_SDK:
        log.info("OpenAI SDK not installed (new interface). Install with `pip install openai` to enable.")
    if not OPENAI_API_KEY:
        log.info("OPENAI_API_KEY not found in environment; OpenAI completions disabled.")

def run_openai_chat(system_prompt: Optional[str], user_prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> (str, Optional[Exception]):
    """Return (output_text, exception_or_none). Uses new SDK: client.chat.completions.create(...)"""
    if openai_client is None:
        return ("", RuntimeError("OpenAI client not available"))
    messages = []
    if system_prompt:
        messages.append({"role":"system", "content": system_prompt})
    messages.append({"role":"user", "content": user_prompt})
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        # new SDK returns objects with .choices[0].message.content
        text = resp.choices[0].message.content.strip()
        return (text, None)
    except Exception as e:
        log.exception("OpenAI request failed")
        return ("", e)

# --- FastAPI app ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {"status":"ok"}

@app.get("/store-info")
async def store_info():
    global _store_meta, _store_emb
    emb_shape = None if _store_emb is None else list(_store_emb.shape)
    return {"meta_count": len(_store_meta), "emb_shape": emb_shape}

@app.post("/ingest-local")
async def ingest_local(filename: str = Form(...)):
    """
    Ingest a local JSON file with structure: list of {id, text, metadata}
    e.g. university_docs.json (the file path may be absolute or relative to project root)
    """
    path = filename
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        return JSONResponse({"status":"error", "detail": f"file not found: {path}"}, status_code=400)

    try:
        data = json.load(open(path, "r", encoding="utf-8"))
        # expected a list of docs with keys id, text, metadata
        docs_to_add = []
        metas_to_add = []
        ids_to_add = []
        # skip ids already present
        existing_ids = set(x["id"] for x in _store_meta)
        for item in data:
            if not isinstance(item, dict) or "id" not in item or "text" not in item:
                continue
            if item["id"] in existing_ids:
                continue
            docs_to_add.append(item["text"])
            metas_to_add.append(item.get("metadata", {}))
            ids_to_add.append(item["id"])
        if docs_to_add:
            embs = embed_model.encode(docs_to_add, show_progress_bar=False).tolist()
            add_documents_to_store(docs_to_add, metas_to_add, ids_to_add, embs)
        return {"status":"ok", "file": path, "chunks": len(docs_to_add)}
    except Exception as e:
        log.exception("ingest-local failed")
        return JSONResponse({"status":"error", "detail": str(e)}, status_code=500)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), course: Optional[str] = Form(None)):
    """Upload a text or pdf file and index into store (will chunk)."""
    fname = f"{uuid.uuid4().hex}_{file.filename}"
    out_path = os.path.join(DATA_DIR, fname)
    with open(out_path, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    text = ""
    try:
        if file.filename.lower().endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(out_path) as pdf:
                for p in pdf.pages:
                    text += (p.extract_text() or "") + "\n"
        else:
            with open(out_path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
    except Exception:
        with open(out_path, "rb") as fh:
            text = fh.read().decode(errors="ignore")

    # chunking
    words = text.split()
    chunk_size = 400
    overlap = 100
    chunks = []
    i = 0
    while i < len(words):
        ch = " ".join(words[i:i+chunk_size])
        if ch.strip():
            chunks.append(ch)
        i += chunk_size - overlap

    if not chunks:
        return JSONResponse({"status":"no_text"}, status_code=400)

    embs = embed_model.encode(chunks, show_progress_bar=False).tolist()
    metas = [{"source": file.filename, "course": course or "general", "chunk_index": idx} for idx in range(len(chunks))]
    ids = [uuid.uuid4().hex for _ in range(len(chunks))]
    add_documents_to_store(chunks, metas, ids, embs)
    return {"status":"ok", "file": file.filename, "chunks": len(chunks)}

@app.post("/query")
async def query_endpoint(request: Request, q: Optional[str] = Form(None), k: Optional[int] = Form(4)):
    """
    Query endpoint accepts:
     - form data (q, k)
     - or JSON body: {"q":"...", "k":4}
    Returns nearest docs from vector store and optionally a generated answer (OpenAI).
    """
    # accept both form and json
    if not q:
        try:
            body = await request.json()
            if isinstance(body, dict):
                q = body.get("q") or q
                k = int(body.get("k", k or 4))
        except Exception:
            # If there's no JSON body, ignore
            pass

    if not q:
        return JSONResponse({"error":"missing_parameter","detail":"Missing 'q' (query text)."}, status_code=422)

    # Build embedding and query store
    q_emb = embed_model.encode([q], show_progress_bar=False).tolist()
    res = query_store(q_emb, n_results=int(k), include=["documents","metadatas","distances"])

    # prepare context for LLM (concatenate top docs)
    top_docs = res["results"][0]["documents"] if res["results"] else []
    context = "\n\n".join(top_docs)
    # limit context characters to avoid huge prompts
    if len(context) > 3500:
        context = context[:3500]

    # If OpenAI client available, call OpenAI; else set used_openai false
    used_openai = False
    answer_text = ""
    openai_err = None
    if openai_client is not None:
        system_prompt = "You are a concise assistant. Use the provided CONTEXT to answer the question. If the answer is not in context, say you don't know."
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{q}\n\nAnswer concisely and cite source ids in parentheses."
        answer_text, openai_err = run_openai_chat(system_prompt, user_prompt, max_tokens=256, temperature=0.0)
        if openai_err is None:
            used_openai = True
        else:
            answer_text = f"OpenAI request failed: {openai_err}"

    # return store results + optional generated answer
    out = {
        "results": res["results"],
        "answer": answer_text,
        "used_openai": used_openai,
        "sources": [
            {
                "id": res["results"][0]["ids"][i],
                "metadata": res["results"][0]["metadatas"][i],
                "distance": res["results"][0]["distances"][i]
            } for i in range(len(res["results"][0]["ids"]))
        ] if res["results"] and res["results"][0]["ids"] else []
    }
    return out

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """Accept audio file, transcribe (if whisper present) and query store / return TTS answer (if available)."""
    uid = uuid.uuid4().hex
    tmpdir = os.path.join(TMP_DIR, uid)
    os.makedirs(tmpdir, exist_ok=True)
    uploaded = os.path.join(tmpdir, "upload")
    with open(uploaded, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    wav = os.path.join(tmpdir, "conv.wav")
    try:
        wav_from_upload(uploaded, wav)
    except Exception as e:
        return JSONResponse({"error": f"ffmpeg failed: {e}"}, status_code=500)

    transcript = ""
    try:
        transcript = run_whisper_on_wav(wav)
    except Exception as e:
        # whisper not available; continue with empty transcript
        log.info(f"Whisper not available or failed: {e}")
        transcript = ""

    q = transcript or ""
    if not q:
        return JSONResponse({"error":"no_transcript","detail":"No transcript produced and no question provided."}, status_code=400)

    q_emb = embed_model.encode([q], show_progress_bar=False).tolist()
    res = query_store(q_emb, n_results=4, include=["documents","metadatas"])
    docs = res["results"][0]["documents"] if res["results"] else []
    context = "\n\n".join(docs)[:3500]

    try:
        prompt = f"Context:\n{context}\n\nQuestion:\n{q}\nAnswer concisely, cite source."
        # use Llama if OpenAI not configured
        if openai_client:
            answer_text, openai_err = run_openai_chat("", prompt, max_tokens=256, temperature=0.0)
            if openai_err:
                answer_text = f"OpenAI failed: {openai_err}"
        else:
            answer_text = run_llama_prompt(prompt, max_tokens=256)
    except Exception as e:
        answer_text = f"LLM failed: {e}"

    resp_wav = os.path.join(tmpdir, "resp.wav")
    audio_b64 = ""
    try:
        tts_say_to_wav(answer_text or "I couldn't find an answer.", resp_wav)
        with open(resp_wav, "rb") as fh:
            audio_b64 = base64.b64encode(fh.read()).decode("utf-8")
    except Exception as e:
        log.info(f"TTS not available or failed: {e}")
        audio_b64 = ""

    return {"transcript": transcript, "answer": answer_text, "audio_base64": audio_b64, "docs": docs}

# EOF
