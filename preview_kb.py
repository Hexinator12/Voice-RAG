# preview_kb.py
import json
from pathlib import Path
p = Path("./voice_rag_kb.json")
if not p.exists():
    raise SystemExit("File not found: /mnt/data/voice_rag_kb.json")

data = json.loads(p.read_text(encoding="utf-8"))

print("Top-level keys and counts:")
for k, v in data.items():
    if isinstance(v, list):
        print(f" - {k}: {len(v)} items")
    else:
        print(f" - {k}: type={type(v).__name__}")

print("\n--- SAMPLE ITEMS (up to 3 each) ---")
for k, v in data.items():
    if isinstance(v, list):
        print(f"\n[{k}]")
        for i, item in enumerate(v[:3]):
            print(f"  {i+1}. {repr(item)[:400]}")
    else:
        print(f"\n[{k}] (full value truncated)\n  {repr(v)[:400]}")
