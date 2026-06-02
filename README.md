# Karpathy Digital Twin
**AIMS DTU Summer Project 2026 — Vedant Khanna**

I built a CLI agent that tries to emulate Andrej Karpathy — not just answer questions about his work, but actually reason the way he does. The idea was that most "digital twin" projects are basically just chatbots with a persona prompt. I wanted to see how far you could push it if you actually thought about the retrieval and reasoning pipeline.

---

## What it does

There are two modes:

**Teaching mode** (default) — ask it anything about ML, neural nets, backprop, transformers, whatever. It answers in Karpathy's voice, grounded in his actual blog posts, lectures, and READMEs rather than just the model's training data.

**Advisory mode (`/think`)** — this is the more interesting one. Give it a real problem you're working on and it thinks through it by connecting the problem to Karpathy's actual project decisions. Like if you ask about architecture choices, it'll reference what he did in nanoGPT and why, or how he approached the same tradeoff in makemore. It's less "here's a generic answer" and more "here's how I'd think about it based on things I've actually built."

```
You: /think I'm building a character level LM, should I start with RNN or go straight to transformer

Karpathy: Let's be honest about what you actually need here. When I was building makemore...
```

---

## The RAG pipeline

I spent most of my time on this. The short version:

- **Chunking is structure-aware** — blogs split on headers, YouTube transcripts split on timestamp gaps (topic shifts), READMEs preserve code blocks. Not fixed word counts, which just randomly slice through ideas.
- **Parent-child chunks** — small chunks for retrieval precision, larger parent chunks sent to the LLM for actual context. Solves the classic RAG tradeoff.
- **HyDE** — instead of searching with the raw user query, I generate a hypothetical Karpathy answer first and search with that. Closes the vocabulary gap between how users ask questions and how Karpathy actually writes.
- **BM25 + semantic search fused with RRF** — neither keyword nor semantic search is strictly better, so run both and merge results via Reciprocal Rank Fusion.
- **Query decomposition** — rewrites the user's question into multiple search queries using Karpathy's vocabulary before retrieval.

## The knowledge graph

For advisory mode I hand-curated a knowledge graph of Karpathy's projects — nanoGPT, micrograd, makemore, llm.c, minbpe. Each node stores the key decisions he made, his reasoning, and lessons learned. Edges connect projects to the concepts they touch and to the cross-project principles he applies everywhere (start simple, measure before optimizing, distrust abstractions until you understand the primitives).

When you use `/think`, the agent traverses this graph to find which projects are relevant to your problem and explicitly reasons from those decisions rather than just doing keyword retrieval.

I could have scraped this automatically but honestly the structured knowledge is what makes advisory mode work — the graph captures the *connections* Karpathy draws between ideas, which vector search just can't do.

## Memory

Short-term is a sliding window of the last N turns, injected into every prompt so the conversation stays coherent. Long-term is SQLite — after each session the agent extracts facts about the user worth remembering (what they're building, what they're stuck on) and stores them. Next session it loads those back in so Karpathy already knows your context.

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
Memory Injection (short + long term)
    ↓
Generation (Gemini 2.5 Flash)
```

---

## Data sources

All scraped from Karpathy's public work:

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
| CLI | rich |
| Knowledge graph | custom, no external DB |

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/AIMS_Karpathy_Twin.git
cd AIMS_Karpathy_Twin
pip install -r requirements.txt
```

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```
Get a free key at https://aistudio.google.com/app/apikey

```bash
python ingest/scraper.py    # scrape data (~5 mins)
python ingest/embedder.py   # build index (~3 mins)
python main.py              # run
```

---

## Commands

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
├── main.py
├── config.py
├── knowledge_graph.py
├── ingest/
│   ├── scraper.py        scrape all sources
│   ├── chunker.py        structure-aware + semantic chunking
│   └── embedder.py       build ChromaDB index
├── rag/
│   ├── retriever.py      BM25 + semantic + RRF
│   ├── rewriter.py       query decomposition + HyDE
│   └── vector_store.py
├── memory/
│   ├── short_term.py
│   └── long_term.py
├── agent/
│   ├── persona.py        system prompt
│   ├── guard.py          persona protection
│   ├── teaching.py       teaching mode
│   └── advisory.py       /think mode
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

I skipped voice cloning — ElevenLabs requires a paid plan and the free alternatives don't sound close enough to be worth the tradeoff. The time was better spent on the RAG pipeline.

The knowledge base JSONs are manually written, not scraped. That was intentional — the point of advisory mode is that it reasons from specific decisions Karpathy made, and you can't really extract that reliably from raw text. It took maybe an hour to write all five and it's what makes `/think` actually useful.

API limits on Gemini's free tier are real. I optimized down to 2 calls per user message (query decomposition + HyDE combined into one, then generation) to stay within them during a demo.

---

*AIMS DTU Summer Project 2026 — Vedant Khanna*
