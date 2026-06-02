from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from config import RAW_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


# ─── Utility ────────────────────────────────────────────────────────────────

def load_raw_docs() -> List[Dict]:
    docs = []
    for path in Path(RAW_DATA_DIR).glob("*"):
        if path.suffix in [".txt", ".md"]:
            text = path.read_text(encoding="utf-8", errors="ignore")
            lines = text.split("\n")
            source = path.name
            if lines[0].startswith("SOURCE:"):
                source = lines[0].replace("SOURCE:", "").strip()
                text = "\n".join(lines[2:])
            docs.append({
                "source": source,
                "content": text,
                "filename": path.name
            })
    print(f"Loaded {len(docs)} raw documents")
    return docs


def detect_source_type(filename: str) -> str:
    if filename.startswith("blog_"):
        return "blog"
    elif filename.startswith("youtube_"):
        return "youtube"
    elif filename.startswith("readme_"):
        return "readme"
    else:
        return "generic"


# ─── Chunking Strategies ─────────────────────────────────────────────────────

def chunk_by_headers(text: str) -> List[str]:
    """Split markdown text on ## or ### headers. Used for blogs + READMEs."""
    sections = re.split(r'\n(?=#{1,3} )', text)
    chunks = []
    for section in sections:
        section = section.strip()
        if len(section.split()) < 20:
            continue  # skip tiny sections
        # if section is still very long, further split by paragraph
        if len(section.split()) > CHUNK_SIZE * 1.5:
            paragraphs = section.split("\n\n")
            buffer = []
            for para in paragraphs:
                buffer.append(para)
                if sum(len(p.split()) for p in buffer) >= CHUNK_SIZE:
                    chunks.append("\n\n".join(buffer))
                    buffer = buffer[-1:]  # keep last para for overlap
            if buffer:
                chunks.append("\n\n".join(buffer))
        else:
            chunks.append(section)
    return chunks


def chunk_by_timestamps(text: str, gap_threshold: int = 30) -> List[str]:
    """
    Split YouTube transcripts on large timestamp gaps (topic shifts).
    Falls back to paragraph splits within long segments.
    """
    lines = text.split("\n")
    segments = []
    current_lines = []
    last_time = 0

    for line in lines:
        match = re.match(r'\[(\d+)s\]', line)
        if match:
            t = int(match.group(1))
            if current_lines and (t - last_time) > gap_threshold:
                segments.append("\n".join(current_lines))
                current_lines = []
            last_time = t
        current_lines.append(line)

    if current_lines:
        segments.append("\n".join(current_lines))

    # further split segments that are too long
    chunks = []
    for seg in segments:
        words = seg.split()
        if len(words) > CHUNK_SIZE:
            for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                chunks.append(" ".join(words[i: i + CHUNK_SIZE]))
        else:
            if len(words) > 20:
                chunks.append(seg)
    return chunks


def chunk_semantic(text: str, threshold: float = 0.45) -> List[str]:
    """
    Semantic chunking — split when cosine similarity between
    consecutive sentences drops (topic boundary detected).
    Used as fallback for generic content.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]

    if len(sentences) < 3:
        return [text] if text.strip() else []

    model = get_model()
    embeddings = model.encode(sentences, show_progress_bar=False)

    chunks = []
    current = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(
            embeddings[i - 1].reshape(1, -1),
            embeddings[i].reshape(1, -1)
        )[0][0]

        current.append(sentences[i])

        # split if topic shifts OR chunk is getting too long
        word_count = sum(len(s.split()) for s in current)
        if (sim < threshold or word_count >= CHUNK_SIZE) and word_count > 30:
            chunks.append(" ".join(current))
            current = [sentences[i]]  # overlap: carry last sentence forward

    if current:
        chunks.append(" ".join(current))

    return chunks


# ─── Parent-Child Builder ─────────────────────────────────────────────────────

def build_parent_child(chunks: List[str], source: str, filename: str) -> List[Dict]:
    """
    For every 3 consecutive child chunks, create a parent chunk.
    Store both. Retrieval uses child, LLM gets parent context.
    """
    results = []
    for i, child_text in enumerate(chunks):
        # parent = this chunk + neighbors for broader context
        start = max(0, i - 1)
        end = min(len(chunks), i + 2)
        parent_text = " ".join(chunks[start:end])

        results.append({
            "id": f"{filename}__child{i}",
            "text": child_text,            # used for retrieval
            "parent_text": parent_text,    # sent to LLM
            "source": source,
            "filename": filename,
            "chunk_index": i,
            "type": "child",
        })
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def chunk_all_docs(docs: List[Dict]) -> List[Dict]:
    all_chunks = []

    for doc in docs:
        source_type = detect_source_type(doc["filename"])
        content = doc["content"]

        print(f"  chunking [{source_type}] {doc['filename']}")

        if source_type in ("blog", "readme"):
            raw_chunks = chunk_by_headers(content)
            # semantic re-split any chunk that's still too long
            final_chunks = []
            for c in raw_chunks:
                if len(c.split()) > CHUNK_SIZE * 1.5:
                    final_chunks.extend(chunk_semantic(c))
                else:
                    final_chunks.append(c)

        elif source_type == "youtube":
            final_chunks = chunk_by_timestamps(content)

        else:
            final_chunks = chunk_semantic(content)

        # build parent-child pairs
        doc_chunks = build_parent_child(
            final_chunks,
            source=doc["source"],
            filename=doc["filename"]
        )
        all_chunks.extend(doc_chunks)

    print(f"\nTotal chunks: {len(all_chunks)} across {len(docs)} documents")
    return all_chunks


if __name__ == "__main__":
    docs = load_raw_docs()
    chunks = chunk_all_docs(docs)
    print(f"\nSample child chunk:\n{chunks[0]['text'][:300]}")
    print(f"\nSample parent chunk:\n{chunks[0]['parent_text'][:500]}")