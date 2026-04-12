"""
Build Index — Pre-compute and save the TF-IDF index for the lessons knowledge base.

Run this once after adding or editing kb_chunks.md:
    python .agents/skills/lessons_rag/scripts/build_index.py
"""

import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_DIR / "scripts"))

from rag_engine import KB_PATH, INDEX_PATH, load_chunks, build_index, save_index


def main():
    print(f"[Build Index] Loading knowledge base from: {KB_PATH}")
    chunks = load_chunks(KB_PATH)
    print(f"[Build Index] Loaded {len(chunks)} chunks.\n")

    for c in chunks:
        print(f"  {c['id']:12s}  {c['category']:30s}  {c['title'][:60]}")

    print("\n[Build Index] Building TF-IDF index...")
    idx = build_index(chunks)
    print(f"[Build Index] Vocabulary size: {len(idx.vocab)} terms | Documents: {idx.n_docs}")

    save_index(idx, INDEX_PATH)
    print("\n[Build Index] Done! Run rag_engine.py to test queries.")


if __name__ == "__main__":
    main()
