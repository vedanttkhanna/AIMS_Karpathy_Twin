import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.chunker import load_raw_docs, chunk_all_docs
from rag.vector_store import add_chunks


def build_index():
    print("=== Building RAG Index ===")
    docs = load_raw_docs()
    chunks = chunk_all_docs(docs)
    add_chunks(chunks)
    print("=== Index Ready ===")


if __name__ == "__main__":
    build_index()
