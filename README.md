# Karpathy Digital Twin
**Vedant Khanna**

I built an agent that tries to emulate Andrej Karpathy — not just answer questions about his work, but actually reason the way he does. Most "digital twin" projects are basically chatbots with a persona prompt. I wanted to see how far you could push it if you actually thought about the retrieval and reasoning pipeline.

Started during summer break after watching a bunch of his videos — wondered what it'd take to build something that doesn't just repeat his lectures back but could actually help think through my own project problems the way he would.

There's both a CLI and a web interface.

---

## What it does

**Teaching mode** (default) — ask it anything about ML, neural nets, backprop, transformers, whatever. Answers in Karpathy's voice, grounded in his actual blog posts, lectures, and READMEs rather than just the model's training data.

**Advisory mode (`/think`)** — give it a real problem you're working on and it thinks through it by connecting the problem to Karpathy's actual project decisions. Reasons less like "here's a generic answer" and more like "here's how I'd think about it based on things I've actually built."

```
You: /think I'm building a character level LM, should I start with RNN or go straight to transformer

Karpathy: Let's be honest about what you actually need here. When I was building makemore...
```

**RL-inspired adaptation** — the web interface tracks implicit feedback from your messages ("I didn't understand that" vs "oh that makes sense") and adjusts explanation depth and style for the rest of the conversation. Not true weight-based RL since the LLM is frozen, but an in-context reward signal that shapes how the twin explains things to you specifically.

---

## The RAG pipeline

- **Structure-aware chunking** — blogs split on headers, YouTube transcripts split on timestamp gaps (topic shifts), READMEs preserve code blocks. Not fixed word counts, which randomly slice through ideas.
- **Parent-child chunks** — small chunks for retrieval precision, larger parent chunks sent to the LLM for context. Solves the classic precision-vs-context RAG tradeoff.
- **HyDE** — generates a hypothetical Karpathy answer first and searches with that instead of the raw query. Closes the vocabulary gap between how users ask questions and how Karpathy actually writes.
- **BM25 + semantic search fused with RRF** — neither keyword nor semantic search is strictly better, so both run and merge via Reciprocal Rank Fusion.
- **Query decomposition** — rewrites the user's question into multiple search queries using Karpathy's vocabulary before retrieval.

## The knowledge graph

For advisory mode I hand-curated a knowledge graph of Karpathy's projects — nanoGPT, micrograd, makemore, llm.c, minbpe. Each node stores the key decisions he made, his reasoning, and lessons learned. Edges connect projects to the concepts they touch and to cross-project principles (start simple, measure before optimizing, distrust abstractions until you understand the primitives).

`/think` first matches keywords from the query to relevant nodes, then traverses two hops outward to surface related projects and principles that aren't explicitly mentioned in the prompt.

The knowledge base is manually written, not scraped — the structured decision+reasoning format is what makes advisory mode actually useful instead of generic.

## Memory

Short-term is a sliding window of the last N turns, injected into every prompt. Long-term is SQLite — after each session the agent extracts facts about the user worth remembering and stores them, loaded back in next session.

---

## Architecture

```
User Query
    ↓
Guard Layer (persona protection)
    ↓
Query Decomposition + HyDE (1 Gemini call)
    ↓
BM25 + Semantic Search → 30 candidates
    ↓
Reciprocal Rank Fusion → top 6 chunks
    ↓
Knowledge Graph Traversal → project context
    ↓
Memory + Adaptation State Injection
    ↓
Generation (Gemini 2.5 Flash)
    ↓
Feedback Analysis → updates Adaptation State for next turn
```

---

## Data sources

- Blog posts — karpathy.github.io (RNN effectiveness, recipe for training, LeCun 1989, hacker's guide to neural nets)
- GitHub READMEs — nanoGPT, micrograd, makemore, llm.c, minbpe
- YouTube transcripts — Zero to Hero series (9 videos): micrograd from scratch, makemore parts 1-5, GPT from scratch, tokenization, GPT-2 reproduction

About 19 documents, ~2100 chunks after chunking.

---

## Stack

| | |
|---|---|
| LLM | Gemini 2.5 Flash |
| Vector store | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 |
| Keyword search | rank-bm25 |
| Memory | SQLite + in-memory buffer |
| Backend | Flask |
| Frontend | HTML/CSS/JS (no framework) |
| Knowledge graph | custom, no external DB |

---

## Setup

```bash
git clone https://github.com/vedanttkhanna/AIMS_Karpathy_Twin.git
cd AIMS_Karpathy_Twin
pip install -r requirements.txt
```

```bash
python ingest/scraper.py    # scrape data (~5 mins)
python ingest/embedder.py   # build index (~3 mins)
```

**Web interface:**
```bash
python app.py
```
Open `http://localhost:5000`, enter your Gemini API key on the login screen (get one free at https://aistudio.google.com/app/apikey).

**CLI:**
```bash
python main.py
```
Requires a `.env` file with `GEMINI_API_KEY=your_key_here`.

---

## CLI Commands

```
/think <problem>   advisory mode — thinks through your problem
/memory            show what the twin remembers about you
/clear             reset conversation
/quit              exit and save session
```

---

## Project structure

```
karpathy-twin/
├── app.py                  Flask backend
├── templates/index.html    web frontend
├── main.py                 CLI entry point
├── config.py
├── knowledge_graph.py
├── ingest/
│   ├── scraper.py          scrape all sources
│   ├── chunker.py          structure-aware + semantic chunking
│   └── embedder.py         build ChromaDB index
├── rag/
│   ├── retriever.py        BM25 + semantic + RRF
│   ├── rewriter.py         query decomposition + HyDE
│   └── vector_store.py
├── memory/
│   ├── short_term.py
│   └── long_term.py
├── agent/
│   ├── persona.py          system prompt
│   ├── guard.py            persona protection
│   ├── teaching.py         teaching mode
│   ├── advisory.py         /think mode
│   └── feedback.py         RL-inspired adaptation loop
└── knowledge_base/
    ├── nanoGPT.json
    ├── micrograd.json
    ├── makemore.json
    ├── llm_c.json
    ├── minbpe.json
    └── principles.json
```

---

## A few notes

Skipped voice cloning — ElevenLabs requires a paid plan and free alternatives don't sound close enough to be worth the tradeoff.

API limits on Gemini's free tier are real. Optimized down to 2 calls per user message (query decomposition + HyDE combined into one, then generation) to stay within them during a demo.

The RL feedback loop is in-context adaptation, not true reinforcement learning — Gemini's weights are frozen, so there's no backpropagation happening. What it does is track a reward signal from implicit sentiment in your messages and inject style instructions into the next prompt accordingly.

---

*Vedant Khanna*
