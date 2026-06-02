import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rank_bm25 import BM25Okapi
from rag.vector_store import semantic_search, get_collection
from config import TOP_K_BM25, TOP_K_SEMANTIC, TOP_K_FINAL
from typing import List, Dict


# BM25 index built lazily from chromadb documents
_bm25 = None
_corpus = None


def build_bm25():
    global _bm25, _corpus
    if _bm25 is not None:
        return
    collection = get_collection()
    results = collection.get(include=["documents", "metadatas"])
    _corpus = [{"text": d, "source": m.get("source",""), "parent_text": m.get("parent_text", d)}
               for d, m in zip(results["documents"], results["metadatas"])]
    tokenized = [doc["text"].lower().split() for doc in _corpus]
    _bm25 = BM25Okapi(tokenized)
    print(f"BM25 index built over {len(_corpus)} chunks")


def bm25_search(query: str, top_k: int = TOP_K_BM25) -> List[Dict]:
    build_bm25()
    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"text": _corpus[i]["parent_text"],
             "retrieval_text": _corpus[i]["text"],
             "source": _corpus[i]["source"],
             "score": float(scores[i]),
             "method": "bm25"} for i in top_indices]


def reciprocal_rank_fusion(semantic: List[Dict], keyword: List[Dict], k: int = 60) -> List[Dict]:
    """Merge semantic + BM25 results using RRF scoring."""
    scores = {}
    docs = {}
    for rank, doc in enumerate(semantic):
        key = doc["text"][:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        docs[key] = doc
    for rank, doc in enumerate(keyword):
        key = doc["text"][:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        if key not in docs:
            docs[key] = doc
    ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [docs[k] for k in ranked[:TOP_K_FINAL]]


def retrieve(query: str) -> List[Dict]:
    semantic = semantic_search(query, top_k=TOP_K_SEMANTIC)
    keyword = bm25_search(query, top_k=TOP_K_BM25)
    fused = reciprocal_rank_fusion(semantic, keyword)
    return fused