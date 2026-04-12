"""
RAG Engine — Portfolio Analyzer Lessons Learned
================================================
Provides TF-IDF based retrieval over the structured knowledge base.
Uses only numpy (already installed) — no scikit-learn required.

Usage (command-line):
    python .agents/skills/lessons_rag/scripts/rag_engine.py "your question here"
    python .agents/skills/lessons_rag/scripts/rag_engine.py "your question" 5   # top 5

Usage (library):
    from rag_engine import query_lessons, format_context
    results = query_lessons("cold start slow startup", top_k=3)
    context = format_context(results)
"""

import math
import os
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SKILL_DIR = Path(__file__).resolve().parent.parent
KB_PATH = SKILL_DIR / "knowledge_base" / "kb_chunks.md"
INDEX_PATH = SKILL_DIR / "knowledge_base" / "tfidf_index.pkl"

# Common English stop words to ignore
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "we", "they", "you", "i", "if", "as",
    "so", "do", "not", "no", "all", "any", "can", "will", "has", "have",
    "had", "s", "e", "g", "etc", "use", "used", "using", "when", "what",
}


# ---------------------------------------------------------------------------
# Chunk parsing
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alpha, remove stop words, return unigrams + bigrams."""
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP_WORDS and len(w) > 1]
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    return words + bigrams


def load_chunks(kb_path: Path = KB_PATH) -> List[Dict[str, Any]]:
    """Parse kb_chunks.md into a list of chunk dicts."""
    text = kb_path.read_text(encoding="utf-8")
    raw_blocks = re.split(r"\n---\n", text)
    chunks = []
    for block in raw_blocks:
        block = block.strip()
        if not block or block.startswith("#"):
            continue
        lines = block.splitlines()
        meta: Dict[str, Any] = {}
        body_start = 0
        for i, line in enumerate(lines):
            if line.startswith("id:"):
                meta["id"] = line.split(":", 1)[1].strip()
            elif line.startswith("category:"):
                meta["category"] = line.split(":", 1)[1].strip()
            elif line.startswith("title:"):
                meta["title"] = line.split(":", 1)[1].strip()
            elif line.startswith("tags:"):
                raw_tags = line.split(":", 1)[1].strip().strip("[]")
                meta["tags"] = [t.strip() for t in raw_tags.split(",") if t.strip()]
            elif line == "" and meta:
                body_start = i + 1
                break
        meta["text"] = "\n".join(lines[body_start:]).strip()
        if "id" in meta and meta["text"]:
            meta["search_text"] = f"{meta.get('title', '')} {' '.join(meta.get('tags', []))} {meta['text']}"
            chunks.append(meta)
    return chunks


# ---------------------------------------------------------------------------
# Pure-numpy TF-IDF
# ---------------------------------------------------------------------------

class TFIDFIndex:
    """Lightweight TF-IDF index using only Python stdlib + numpy."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: List[float] = []
        self.doc_vectors: Optional[Any] = None  # numpy array shape (n_docs, vocab_size)
        self.n_docs: int = 0

    def fit(self, corpus: List[str]):
        import numpy as np

        tokenized = [tokenize(doc) for doc in corpus]
        self.n_docs = len(tokenized)

        # Build vocabulary
        vocab_set: set = set()
        for tokens in tokenized:
            vocab_set.update(tokens)
        self.vocab = {term: idx for idx, term in enumerate(sorted(vocab_set))}
        V = len(self.vocab)

        # Doc-freq count for IDF
        df = [0] * V
        for tokens in tokenized:
            seen = set(tokens)
            for t in seen:
                if t in self.vocab:
                    df[self.vocab[t]] += 1

        # IDF with smoothing: log((1+N)/(1+df)) + 1
        self.idf = [
            math.log((1 + self.n_docs) / (1 + df[i])) + 1 for i in range(V)
        ]

        # Build doc matrix (TF * IDF, L2-normalised)
        mat = np.zeros((self.n_docs, V), dtype=np.float32)
        for d, tokens in enumerate(tokenized):
            tf = Counter(tokens)
            total = len(tokens)
            for term, count in tf.items():
                if term in self.vocab:
                    j = self.vocab[term]
                    mat[d, j] = (1 + math.log(count)) * self.idf[j]

        # L2 normalise each row
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_vectors = mat / norms

    def query(self, question: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Return (doc_index, cosine_score) pairs, sorted descending."""
        import numpy as np

        tokens = tokenize(question)
        V = len(self.vocab)
        q_vec = np.zeros(V, dtype=np.float32)
        tf = Counter(tokens)
        for term, count in tf.items():
            if term in self.vocab:
                j = self.vocab[term]
                q_vec[j] = (1 + math.log(count)) * self.idf[j]

        norm = np.linalg.norm(q_vec)
        if norm == 0:
            return []
        q_vec /= norm

        scores = self.doc_vectors @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------

def build_index(chunks: List[Dict[str, Any]]) -> TFIDFIndex:
    corpus = [c["search_text"] for c in chunks]
    idx = TFIDFIndex()
    idx.fit(corpus)
    return idx


def save_index(idx: TFIDFIndex, path: Path = INDEX_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(idx, f)
    print(f"[RAG] Index saved → {path}")


def load_index(path: Path = INDEX_PATH) -> Optional[TFIDFIndex]:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_or_build_index(chunks: List[Dict[str, Any]], index_path: Path = INDEX_PATH) -> TFIDFIndex:
    idx = load_index(index_path)
    if idx is None:
        idx = build_index(chunks)
        save_index(idx, index_path)
    return idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_lessons(
    question: str, top_k: int = 3, kb_path: Path = KB_PATH, index_path: Path = INDEX_PATH
) -> List[Dict[str, Any]]:
    """
    Retrieve the top_k most relevant lesson chunks for a question.

    Returns a list of dicts, each with:
        id, category, title, tags, text, score, formatted
    """
    chunks = load_chunks(kb_path)
    idx = _get_or_build_index(chunks, index_path)
    hits = idx.query(question, top_k=top_k)
    results = []
    for doc_idx, score in hits:
        c = chunks[doc_idx]
        formatted = (
            f"### [{c['id']}] {c['title']}\n"
            f"**Category**: {c['category']} | **Score**: {score:.3f}\n\n"
            f"{c['text']}"
        )
        results.append({**c, "score": score, "formatted": formatted})
    return results


def format_context(results: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a single context block for inclusion in a prompt."""
    if not results:
        return "No relevant lessons found."
    parts = [
        "## Relevant Lessons from Project Knowledge Base\n",
        "> The following lessons were retrieved from the portfolio analyzer's documented experience.\n",
    ]
    for r in results:
        parts.append(r["formatted"])
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag_engine.py \"your question here\" [top_k]")
        sys.exit(1)

    question = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"\n🔍 Query: {question}\n{'=' * 60}")
    results = query_lessons(question, top_k=top_k)
    if not results:
        print("No results found (all scores were zero — try different keywords).")
    else:
        for r in results:
            print(f"\n  [{r['id']}]  {r['title']}")
            print(f"  Category : {r['category']}")
            print(f"  Score    : {r['score']:.3f}")
            print(f"  Tags     : {', '.join(r['tags'])}")
            excerpt = r["text"][:400].replace("\n", " ")
            print(f"\n  {excerpt}...\n")
