import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from config import GROQ_MODEL
from core.key_manager import key_manager


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
        return (self.satisfaction_count * 1.0) - (self.confusion_count * 1.5)


def analyze_feedback(user_message: str, previous_response: str) -> dict:
    prompt = f"""You are analyzing a user's message to detect implicit feedback about a previous AI response.

Previous AI response (summary): {previous_response[:300]}...

User's follow-up message: {user_message}

Classify this message and return ONLY a JSON object with these exact fields:
- "is_feedback": true if this message is reacting to the previous response, false if it's a new question
- "sentiment": one of: "confused", "satisfied", "wants_more_depth", "wants_simpler", "engaged", "neutral"
- "confidence": float between 0.0 and 1.0
- "adjustment": one sentence describing how the next response should differ (or "none" if neutral/new question)

Return only valid JSON, no markdown."""

    try:
        response = key_manager.generate_content(
            contents=prompt,
            model=GROQ_MODEL
        )
        text = response.text.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[feedback] error: {e}")

    return {
        "is_feedback": False,
        "sentiment": "neutral",
        "confidence": 0.0,
        "adjustment": "none"
    }