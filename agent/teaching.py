import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from agent.persona import KARPATHY_SYSTEM_PROMPT
from rag.rewriter import smart_retrieve
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from config import GEMINI_API_KEY, GEMINI_MODEL
from typing import List, Dict


def format_context(chunks: List[Dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(f"[Source {i+1}: {chunk.get('source', 'unknown')}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate_response(
    query: str,
    short_term: ShortTermMemory,
    long_term: LongTermMemory,
    session_id: str
) -> str:
    from core.knowledge_graph import get_kg

    # retrieval
    chunks = smart_retrieve(query, short_term.get())
    context = format_context(chunks)
    lt_context = long_term.format_for_prompt()

    # knowledge graph context
    kg = get_kg()
    kg_context = kg.format_for_prompt(query)

    prompt = f"""{KARPATHY_SYSTEM_PROMPT}

## What you remember about this user
{lt_context if lt_context else "New user."}

## Knowledge graph context
{kg_context if kg_context else ""}

## Retrieved context from your work
{context}

## Conversation so far
{short_term.format_for_prompt()}

## Question
{query}

Answer as Andrej Karpathy. Be specific, reference the context above where relevant,
and stay true to your teaching style. If the context doesn't cover the question well,
answer from your general knowledge but be honest about uncertainty."""

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    answer = response.text.strip()

    # extract and save user facts worth remembering
    _extract_user_facts(query, long_term, session_id)

    return answer


def _extract_user_facts(query: str, long_term: LongTermMemory, session_id: str):
    """Use LLM to decide if there's anything worth remembering about the user."""
    prompt = f"""Given this user message, extract any personal context worth remembering 
about the user (what they're building, learning, struggling with, their background).
If there's nothing worth remembering, return exactly: NOTHING
If there is, return a single concise sentence starting with "User is..." or "User wants..."

User message: {query}"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        result = response.text.strip()
        if result and result != "NOTHING":
            long_term.save_fact(session_id, result, type="user_context")
    except Exception as e:
        pass  # non-critical, don't break the main flow