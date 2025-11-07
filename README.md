# Voice-RAG

Local, lightweight Voice-enabled RAG playground with FastAPI and a simple vector store.

## Features
- FastAPI backend with endpoints for ingest, query, and audio upload
- Sentence-Transformers embeddings (`all-MiniLM-L6-v2`)
- Local vector store persisted as numpy + pickle
- Optional OpenAI chat completions (new OpenAI SDK) or local llama.cpp fallback
- Utilities to turn a source JSON into chunks and build FAISS/Chroma indexes
- Simple React frontend (Vite) with text chat and microphone recording

## Repo layout
- `backend/main.py` FastAPI app (local store, optional OpenAI/llama.cpp, audio helpers)
- `backend/app_rag.py` Alternative FAISS-backed API
- `index_faiss.py` Build FAISS index from `university_docs.json`
- `query_faiss.py` Query FAISS index from CLI
- `index_chroma.py` Build Chroma DB from `university_docs.json`
- `transform_and_chunk.py` Convert `voice_rag_kb.json` into chunked `university_docs.json`

Note: Large/local artifacts are ignored via `.gitignore` (e.g., `venv/`, `backend/models/`, `backend/tmp/`, `llama.cpp/`, `whisper.cpp/`, media files, `.env`).

## Requirements
Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional system tools (for audio helpers):
- ffmpeg (for audio conversion)
- macOS `say` TTS is used in examples; replace with your own TTS on other OSes

## Environment
Create a `.env` file in repo root (not committed) with:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

If `OPENAI_API_KEY` is missing, endpoints will still work but LLM answers will use local fallback or be disabled.

## Run the backend (local store)
Start the main FastAPI app:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:
- GET `http://localhost:8000/health` -> `{ "status": "ok" }`

### Ingest a local JSON file
- POST `http://localhost:8000/ingest-local`
  - Form data: `filename=/absolute/or/relative/path/to/university_docs.json`

### Upload and index a file (txt/pdf)
- POST `http://localhost:8000/upload`
  - Form fields: `file=<file>`, `course=<optional>`

### Query
- POST `http://localhost:8000/query`
  - Form: `q`, `k`
  - Or JSON: `{ "q": "...", "k": 4 }`
  - Returns nearest docs and optional LLM answer

### Audio query (optional tools required)
- POST `http://localhost:8000/upload-audio`
  - Form: `file=<audio>`
  - Converts to wav (ffmpeg), transcribes with `whisper-cli` if present, answers and returns TTS wav base64

## Frontend (Vite + React)
A minimal chat UI with text input and microphone recording lives under `frontend/`.

Run the dev server:

```bash
cd frontend
npm install
npm run dev
```

Then open the printed Vite URL (e.g., http://localhost:5173). The app expects the backend at `http://localhost:8000`. If you need to change it, edit the fetch URLs in `frontend/src/app.jsx`.

What it can do:
- Send text questions to `POST /query`
- Record audio via mic, send to `POST /upload-audio`, and auto-play the TTS answer if provided

## Build utilities
- Transform raw KB to chunks:

```bash
python transform_and_chunk.py
```

- Build FAISS index:

```bash
python index_faiss.py
```

- Query FAISS from CLI:

```bash
python query_faiss.py
```

- Build Chroma DB:

```bash
python index_chroma.py
```

## Optional local LLM / STT
- Place GGUF model at `llama.cpp/models/<your-model>.gguf` and build `llama.cpp` (ignored in repo)
- Place Whisper model at `backend/models/ggml-small.bin` and build `whisper.cpp` (ignored in repo)

## Notes
- Do not commit large artifacts or secrets. `.env`, audio, models, tmp, venv are ignored.
- If you want to track `llama.cpp`/`whisper.cpp`, prefer Git submodules.
