import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL
from config import GEMINI_API_KEY, GEMINI_MODEL
from rag.retriever import retrieve
from typing import List, Dict
from rag.vector_store import semantic_search



def rewrite_query(query: str, conversation_history: List[Dict] = []) -> List[str]:
    """
    Decompose + rewrite query into multiple search queries
    using Karpathy's vocabulary and framing.
    """
    history_text = ""
    if conversation_history:
        history_text = "\n".join([f"{m['role']}: {m['content']}"
                                   for m in conversation_history[-4:]])

    prompt = f"""You are helping retrieve information from Andrej Karpathy's work.
Given this user question, generate 3 different search queries that would help find relevant information.
Use Karpathy's specific vocabulary and concepts (micrograd, nanoGPT, makemore, loss landscape, etc).
Return ONLY a JSON array of 3 strings, nothing else.

Conversation context:
{history_text}

User question: {query}

Return format: ["query1", "query2", "query3"]"""

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
   )
    text = response.text.strip()

    import json, re
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        queries = json.loads(match.group())
        return [query] + queries  # original + rewrites
    return [query]


def hyde_retrieve(query: str) -> List[Dict]:
    """
    HyDE: generate a hypothetical Karpathy answer,
    use that to retrieve instead of the raw query.
    """
    prompt = f"""Write a short paragraph (4-5 sentences) as if Andrej Karpathy 
is answering this question in his typical teaching style:

{query}

Be technical, use his vocabulary, reference his projects where relevant."""

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    hypothetical_answer = response.text.strip()
    # retrieve using the hypothetical answer as query
    return retrieve(hypothetical_answer)

def smart_retrieve(query: str, conversation_history=None) -> List[Dict]:
    if conversation_history is None:
        conversation_history = []

    history_text = ""
    if conversation_history:
        history_text = "\n".join([
            f"{m['role']}: {m['content']}" 
            for m in conversation_history[-4:]
        ])

    # single call that does both rewriting AND HyDE
    prompt = f"""You are helping retrieve information from Andrej Karpathy's work.
Given the user question, return a JSON object with exactly two keys:
1. "queries": array of 3 search queries using Karpathy's vocabulary (micrograd, nanoGPT, loss curves, etc.)
2. "hypothetical": a 3-sentence paragraph written AS Karpathy answering the question

Return ONLY valid JSON, no markdown, no explanation.

Conversation context: {history_text}
User question: {query}

Format: {{"queries": ["q1", "q2", "q3"], "hypothetical": "paragraph here"}}"""

    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        import json, re
        text = response.text.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            queries = [query] + data.get("queries", [])
            hypothetical = data.get("hypothetical", "")
        else:
            queries = [query]
            hypothetical = ""
    except Exception as e:
        print(f"[rewriter] fallback to raw query: {e}")
        queries = [query]
        hypothetical = ""

    # retrieve using all queries + hypothetical
    from rag.retriever import reciprocal_rank_fusion
    all_results = []
    for q in queries[:3]:
        all_results.extend(semantic_search(q, top_k=6))
    if hypothetical:
        all_results.extend(semantic_search(hypothetical, top_k=6))

    # deduplicate
    seen = set()
    unique = []
    for doc in all_results:
        key = doc["text"][:80]
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    unique.sort(key=lambda x: x.get("score", 0), reverse=True)
    return unique[:6]