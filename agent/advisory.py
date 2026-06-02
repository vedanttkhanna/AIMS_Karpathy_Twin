import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from google import genai
from agent.persona import KARPATHY_SYSTEM_PROMPT
from rag.rewriter import smart_retrieve
from memory.short_term import ShortTermMemory
from config import GEMINI_API_KEY, GEMINI_MODEL, KNOWLEDGE_BASE_DIR
from core.knowledge_graph import get_kg


def load_knowledge_base() -> str:
    kb_parts = []
    kb_path = Path(KNOWLEDGE_BASE_DIR)
    for json_file in kb_path.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            kb_parts.append(f"### {json_file.stem}\n{json.dumps(data, indent=2)}")
        except Exception:
            pass
    return "\n\n".join(kb_parts)


def think(query: str, short_term: ShortTermMemory) -> str:
    """
    Advisory mode — Karpathy thinks through a practical problem
    using knowledge graph + RAG + Gemini.
    """
    # get knowledge graph context
    kg = get_kg()
    kg_context = kg.format_for_prompt(query)

    # get RAG context
    rag_chunks = smart_retrieve(query, short_term.get())
    rag_context = "\n\n".join(c["text"] for c in rag_chunks[:4])

    prompt = f"""{KARPATHY_SYSTEM_PROMPT}

## Your Project Knowledge Base (from knowledge graph)
{kg_context if kg_context else "No directly related projects found."}

## Relevant context from your writing and lectures
{rag_context}

## Conversation so far
{short_term.format_for_prompt()}

## Problem to think through
{query}

Think through this as Andrej Karpathy:
1. Reduce to simplest form — what is the core of this problem?
2. Connect to your projects — which of your builds touched something similar?
   Reference specific decisions you made and why.
3. Apply your principles — start simple, measure first, distrust complexity
4. Give a concrete recommendation with a clear next step

Show the reasoning, then land on a specific actionable answer.
Reference your actual projects naturally, not as a list."""

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text.strip()