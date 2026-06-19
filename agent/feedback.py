import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL


class AdaptationState:
    def __init__(self):
        self.depth_level = 0          # -2 simple ←→ +2 technical
        self.confusion_count = 0
        self.satisfaction_count = 0
        self.style_notes = []         # rolling log of adjustments

    def update(self, sentiment: str, adjustment: str):
        if sentiment == "confused":
            self.depth_level = max(-2, self.depth_level - 1)
            self.confusion_count += 1
            self.style_notes.append("user was confused — use more analogies, less jargon")

        elif sentiment == "wants_simpler":
            self.depth_level = max(-2, self.depth_level - 1)
            self.style_notes.append("user wants simpler explanation")

        elif sentiment == "wants_more_depth":
            self.depth_level = min(2, self.depth_level + 1)
            self.style_notes.append("user wants more technical depth")

        elif sentiment == "satisfied":
            self.satisfaction_count += 1
            self.style_notes.append("user understood — maintain this style")

        elif sentiment == "engaged":
            self.satisfaction_count += 1
            self.style_notes.append("user is engaged and following along")

        # keep only last 5 signals to avoid prompt bloat
        self.style_notes = self.style_notes[-5:]

    def format_for_prompt(self) -> str:
        if not self.style_notes:
            return ""

        depth_instruction = ""
        if self.depth_level <= -1:
            depth_instruction = "This user needs simpler explanations. Use more analogies. Avoid heavy math notation."
        elif self.depth_level == 0:
            depth_instruction = "Balanced depth — intuition first, then technical detail."
        elif self.depth_level >= 1:
            depth_instruction = "This user wants technical depth. Don't dumb it down."

        return f"""## What you've learned about this user so far
{depth_instruction}
Signals from their reactions: {'; '.join(self.style_notes[-3:])}
Confusion count: {self.confusion_count} | Satisfaction count: {self.satisfaction_count}
Adjust your response style accordingly."""

    def get_reward(self) -> float:
        """
        Scalar reward signal for the conversation so far.
        Positive = doing well, negative = losing the user.
        Not used for weight updates (frozen LLM) but logged
        and used to modulate response verbosity and style.
        """
        return (self.satisfaction_count * 1.0) - (self.confusion_count * 1.5)


def analyze_feedback(user_message: str, previous_response: str) -> dict:
    """
    Given the user's follow-up message and the previous response,
    classify whether this is implicit feedback or a new question,
    and if feedback, what signal it carries.
    """
    prompt = f"""You are analyzing a user's message to detect implicit feedback about a previous AI response.

Previous AI response (summary): {previous_response[:300]}...

User's follow-up message: {user_message}

Classify this message and return ONLY a JSON object with these exact fields:
- "is_feedback": true if this message is reacting to the previous response, false if it's a new question
- "sentiment": one of: "confused", "satisfied", "wants_more_depth", "wants_simpler", "engaged", "neutral"
- "confidence": float between 0.0 and 1.0
- "adjustment": one sentence describing how the next response should differ (or "none" if neutral/new question)

Examples of feedback signals:
- "I didn't understand that" → confused
- "oh that makes sense!" → satisfied  
- "can you go deeper on X" → wants_more_depth
- "can you explain that more simply" → wants_simpler
- "interesting, what about Y" → engaged
- "what is X" (new question, no reaction) → is_feedback: false

Return only valid JSON, no markdown."""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        text = response.text.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        pass

    # fallback — treat as neutral new question
    return {
        "is_feedback": False,
        "sentiment": "neutral",
        "confidence": 0.0,
        "adjustment": "none"
    }