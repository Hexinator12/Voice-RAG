# transform_and_chunk.py
import json, uuid, re
from pathlib import Path
from datetime import datetime

SRC = Path("./voice_rag_kb.json")
OUT = Path("./university_docs.json")
MAX_CHARS = 3000            # chunk size (characters). Adjust if you plan token-based split.
OVERLAP_CHARS = 400         # overlap between chunks

def clean_text(s):
    if s is None:
        return ""
    # remove repeated whitespace, weird characters, simple HTML tags
    s = re.sub(r"<[^>]+>", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_text_from_item(key, item):
    # Produce a readable textual blob from each item depending on type
    if isinstance(item, dict):
        parts = []
        # keep ordering for human readability
        for k, v in item.items():
            if v is None or v == "":
                continue
            # if list, join; if dict, flatten a bit
            if isinstance(v, list):
                parts.append(f"{k}: " + ", ".join(map(str, v)))
            elif isinstance(v, dict):
                parts.append(f"{k}: " + ", ".join(f'{kk}={vv}' for kk, vv in v.items()))
            else:
                parts.append(f"{k}: {v}")
        return " | ".join(parts)
    return str(item)

def chunk_text(text, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS):
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def main():
    assert SRC.exists(), f"Source not found: {SRC}"
    data = json.loads(SRC.read_text(encoding="utf-8"))
    docs = []
    meta = data.get("meta", {})
    created_on = meta.get("created_on", datetime.utcnow().isoformat())

    for topk, value in data.items():
        if topk == "meta":
            continue
        if isinstance(value, list):
            for idx, item in enumerate(value):
                text = to_text_from_item(topk, item)
                if not text:
                    continue
                chunks = chunk_text(text)
                for ci, c in enumerate(chunks):
                    docs.append({
                        "id": str(uuid.uuid4()),
                        "text": c,
                        "metadata": {
                            "source": topk,
                            "original_index": idx,
                            "chunk_index": ci,
                            "created_on": created_on
                        }
                    })
        else:
            # In case there are single objects
            text = to_text_from_item(topk, value)
            if text:
                for ci, c in enumerate(chunk_text(text)):
                    docs.append({
                        "id": str(uuid.uuid4()),
                        "text": c,
                        "metadata": {
                            "source": topk,
                            "original_index": 0,
                            "chunk_index": ci,
                            "created_on": created_on
                        }
                    })

    OUT.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(docs)} doc chunks to {OUT}")

if __name__ == "__main__":
    main()
