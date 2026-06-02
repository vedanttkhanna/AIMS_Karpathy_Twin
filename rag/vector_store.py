import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from config import CHROMA_DIR, EMBEDDING_MODEL
import os

os.makedirs(CHROMA_DIR, exist_ok=True)

# singleton pattern so we don't reload the model on every call
_model = None
_client = None
_collection = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name="karpathy",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed(texts: List[str]) -> List[List[float]]:
    model = get_model()
    return model.encode(texts, show_progress_bar=False).tolist()


def add_chunks(chunks: List[Dict]):
    """Embed and insert chunks into ChromaDB."""
    collection = get_collection()

    # skip if already populated
    if collection.count() > 0:
        print(f"Vector store already has {collection.count()} chunks, skipping insert.")
        return

    batch_size = 128
    print(f"Embedding and inserting {len(chunks)} chunks...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [{"source": c["source"],"filename": c["filename"],"parent_text":
                       c.get("parent_text", c["text"])} for c in batch]
        embeddings = embed(texts)
        collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
        print(f"  inserted {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print(f"Vector store ready with {collection.count()} chunks.")


def semantic_search(query: str, top_k: int = 15) -> List[Dict]:
    """Return top_k semantically similar chunks."""
    collection = get_collection()
    query_embedding = embed([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": meta.get("parent_text", doc),  # LLM gets parent context
            "retrieval_text": doc,                  # what was actually matched
            "source": meta.get("source", ""),
            "score": 1 - dist,
       })
    return chunks