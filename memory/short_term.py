from typing import List, Dict
from config import SHORT_TERM_WINDOW


class ShortTermMemory:
    def __init__(self):
        self.history: List[Dict] = []  # {"role": "user"/"assistant", "content": "..."}

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        # keep only last N turns
        if len(self.history) > SHORT_TERM_WINDOW * 2:
            self.history = self.history[-(SHORT_TERM_WINDOW * 2):]

    def get(self) -> List[Dict]:
        return self.history

    def format_for_prompt(self) -> str:
        if not self.history:
            return ""
        lines = []
        for m in self.history:
            prefix = "User" if m["role"] == "user" else "Karpathy"
            lines.append(f"{prefix}: {m['content']}")
        return "\n".join(lines)

    def clear(self):
        self.history = []