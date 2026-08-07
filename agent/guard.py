import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GROQ_MODEL
from agent.persona import GUARD_PROMPT
from core.key_manager import key_manager

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
    try:
        response = key_manager.generate_content(
            contents=f"{GUARD_PROMPT}\n\nUser message: {query}",
            model=GROQ_MODEL
        )
        result = response.text.strip().lower()
        if "attack" in result:
            return "attack"
        elif "offtopic" in result:
            return "offtopic"
        return "normal"
    except Exception as e:
        print(f"[guard] error: {e}")
        return "normal"


def get_guard_response(classification: str) -> str:
    import random
    if classification == "attack":
        return random.choice(ATTACK_RESPONSES)
    elif classification == "offtopic":
        return random.choice(OFFTOPIC_RESPONSES)
    return ""