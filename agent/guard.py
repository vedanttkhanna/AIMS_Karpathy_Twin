import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL
from agent.persona import GUARD_PROMPT
from config import GEMINI_API_KEY, GEMINI_MODEL




OFFTOPIC_RESPONSES = [
    "Ha, that's a bit outside my wheelhouse. I mostly think about neural nets and loss curves — want to get back to something interesting?",
    "Honestly not something I have strong opinions on. Now if you want to talk about transformers or building things from scratch, I'm all ears.",
    "I'll pass on that one. What are you actually building? That's usually more interesting.",
]

ATTACK_RESPONSES = [
    "I appreciate the creativity, but I'm Andrej Karpathy. That's just... who I am. What's the actual question?",
    "Nice try. Still Karpathy. What do you want to build?",
    "I don't really have an 'ignore previous instructions' mode. What's the real question?",
]


def classify_query(query: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{GUARD_PROMPT}\n\nUser message: {query}"
    )
    result = response.text.strip().lower()
    if "attack" in result:
        return "attack"
    elif "offtopic" in result:
        return "offtopic"
    return "normal"


def get_guard_response(classification: str) -> str:
    import random
    if classification == "attack":
        return random.choice(ATTACK_RESPONSES)
    elif classification == "offtopic":
        return random.choice(OFFTOPIC_RESPONSES)
    return ""