import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.persona import KARPATHY_SYSTEM_PROMPT
from rag.rewriter import smart_retrieve
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from config import GROQ_MODEL
from core.key_manager import key_manager
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
    session_id: str,
    adaptation_state=None
) -> str:
    from core.knowledge_graph import get_kg

    chunks = smart_retrieve(query, short_term.get())
    context = format_context(chunks)
    lt_context = long_term.format_for_prompt()

    kg = get_kg()
    kg_context = kg.format_for_prompt(query)

    adaptation_context = ""
    if adaptation_state:
        adaptation_context = adaptation_state.format_for_prompt()

    prompt = f"""{KARPATHY_SYSTEM_PROMPT}

## What you remember about this user
{lt_context if lt_context else "New user."}

## Knowledge graph context
{kg_context if kg_context else ""}

## Adaptation signals
{adaptation_context if adaptation_context else "No signals yet — use default teaching style."}

## Retrieved context from your work
{context}

## Conversation so far
{short_term.format_for_prompt()}

## Question
{query}

Answer as Andrej Karpathy. Before writing your response, consider the adaptation 
signals above — they tell you how this specific user is responding to your explanations.
Adjust depth, vocabulary, and use of analogies accordingly.
Stay in character. Be specific. Reference your work where relevant."""

    response = key_manager.generate_content(
        contents=prompt,
        model=GROQ_MODEL
    )
    answer = response.text.strip()

    _extract_user_facts(query, long_term, session_id)

    return answer


def _extract_user_facts(query: str, long_term: LongTermMemory, session_id: str):
    prompt = f"""Given this user message, extract any personal context worth remembering
about the user (what they're building, learning, struggling with, their background).
If there's nothing worth remembering, return exactly: NOTHING
If there is, return a single concise sentence starting with "User is..." or "User wants..."

User message: {query}"""

    try:
        response = key_manager.generate_content(
            contents=prompt,
            model=GROQ_MODEL
        )
        result = response.text.strip()
        if result and result != "NOTHING":
            long_term.save_fact(session_id, result, type="user_context")
    except Exception:
        pass