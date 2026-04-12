---
name: Lessons Learned RAG (Knowledge Base Retriever)
description: A TF-IDF retrieval system that finds the most relevant lessons learned during portfolio analyzer development. Query it before debugging, designing new features, or deploying to instantly surface documented experience (30+ lessons across Architecture, Performance/DB, UI/UX, Deployment, AI Assistant categories).
---

# Lessons Learned RAG — Portfolio Analyzer Knowledge Base

This skill provides a **Retrieval-Augmented Generation (RAG)** system over all documented lessons learned while building the Portfolio Analyzer. Before investigating a bug, adding a feature, or deploying, query the knowledge base to surface directly relevant past experience.

## What the Knowledge Base Covers

| Category | # Chunks | Topics |
|---|---|---|
| **Architecture** | 5 | Cold start, env identity, intent routing, LLM bypass, new-intent checklist |
| **Performance & DB** | 4 | Composite index, N+1 queries, Neon 500MB limit, sector partial-match filter |
| **UI/UX** | 3 | Immediate/Optional pattern, session-state cache, stale date forcing |
| **Deployment & GCP** | 6 | Memory quota, multiprocessing disconnections, Windows Unicode crash, domain mapping, Secret Manager, OAuth redirect |
| **AI Assistant** | 7 | Intent prompt ordering, regex param extraction, anti-hallucination, sector enrichment, JSON fallback, deterministic override, rank semantics |
| **Maintenance** | 2 | DB sync checklist, full deployment checklist |

## How to Query

### Command-Line (preferred for agent use)

```bash
# From the portfolio_analyzer root directory:
.venv\Scripts\python .agents/skills/lessons_rag/scripts/rag_engine.py "your question" [top_k]

# Examples:
.venv\Scripts\python .agents/skills/lessons_rag/scripts/rag_engine.py "cold start slow startup" 3
.venv\Scripts\python .agents/skills/lessons_rag/scripts/rag_engine.py "sector ranking wrong intent classified as market overview"
.venv\Scripts\python .agents/skills/lessons_rag/scripts/rag_engine.py "neon database connection child process multiprocessing"
.venv\Scripts\python .agents/skills/lessons_rag/scripts/rag_engine.py "stale date session state cache"
.venv\Scripts\python .agents/skills/lessons_rag/scripts/rag_engine.py "how to add a new intent to the assistant"
```

### Python Import (for embedding in prompts or scripts)

```python
import sys
sys.path.insert(0, ".agents/skills/lessons_rag/scripts")
from rag_engine import query_lessons, format_context

results = query_lessons("sector routing misclassification", top_k=3)
context = format_context(results)   # ready-to-paste markdown context block
print(context)
```

## Rebuilding the Index

The TF-IDF index is pre-built and saved at `knowledge_base/tfidf_index.pkl`. If you add or edit chunks in `kb_chunks.md`, rebuild the index:

```bash
.venv\Scripts\python .agents/skills/lessons_rag/scripts/build_index.py
```

## Reading Results

Each result shows:
- **ID** (e.g., `ARCH-003`) — category prefix + sequence number
- **Title** — what the lesson is about
- **Score** — cosine similarity (0–1, higher = more relevant)
- **Text** — Problem, Lesson, and Fix Pattern

A score above **0.15** is generally actionable. Multiple results with similar scores are equally worth reading.

## When to Use

| Situation | Example Query |
|---|---|
| Debugging a slow page or API | `"page is slow cold start"` |
| Adding a new assistant intent | `"how to add a new intent"` |
| Fixing stale / wrong UI data | `"stale date cache session"` |
| Deploying to Cloud Run | `"memory quota deployment cloud run"` |
| LLM returning wrong intent | `"llm misclassifying intent market overview sector"` |
| DB query returning nothing | `"sector filter no results partial match"` |
| Multiprocessing with Neon | `"postgres connection child process fork"` |

## Knowledge Base Location

`knowledge_base/kb_chunks.md` — Plain markdown, human-readable. Add new lessons here in the same `---frontmatter--- / body` format.
