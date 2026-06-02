import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import json
from datetime import datetime
from typing import List, Dict
from config import MEMORY_DB_PATH


class LongTermMemory:
    def __init__(self):
        os.makedirs(os.path.dirname(MEMORY_DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(MEMORY_DB_PATH, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                type TEXT,
                content TEXT,
                metadata TEXT,
                created_at TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                summary TEXT,
                created_at TEXT,
                last_active TEXT
            )
        """)
        self.conn.commit()

    def save_fact(self, session_id: str, content: str, type: str = "fact"):
        """Save something the user mentioned worth remembering."""
        self.conn.execute(
            "INSERT INTO memories (session_id, type, content, metadata, created_at) VALUES (?,?,?,?,?)",
            (session_id, type, content, "{}", datetime.now().isoformat())
        )
        self.conn.commit()

    def save_session_summary(self, session_id: str, summary: str):
        self.conn.execute("""
            INSERT INTO sessions (id, summary, created_at, last_active)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET summary=excluded.summary, last_active=excluded.last_active
        """, (session_id, summary, datetime.now().isoformat(), datetime.now().isoformat()))
        self.conn.commit()

    def get_recent_sessions(self, limit: int = 3) -> List[Dict]:
        cursor = self.conn.execute(
            "SELECT id, summary, last_active FROM sessions ORDER BY last_active DESC LIMIT ?",
            (limit,)
        )
        return [{"session_id": r[0], "summary": r[1], "last_active": r[2]}
                for r in cursor.fetchall()]

    def get_all_facts(self) -> List[str]:
        cursor = self.conn.execute(
            "SELECT content FROM memories ORDER BY created_at DESC LIMIT 20"
        )
        return [r[0] for r in cursor.fetchall()]

    def format_for_prompt(self) -> str:
        facts = self.get_all_facts()
        sessions = self.get_recent_sessions()
        parts = []
        if facts:
            parts.append("Things I remember about this user:\n" + "\n".join(f"- {f}" for f in facts))
        if sessions:
            parts.append("Recent conversations:\n" + "\n".join(
                f"- [{s['last_active'][:10]}] {s['summary']}" for s in sessions))
        return "\n\n".join(parts)

    def show(self):
        """For /memory command in CLI."""
        facts = self.get_all_facts()
        sessions = self.get_recent_sessions(5)
        print("\n=== Long Term Memory ===")
        print(f"Facts remembered: {len(facts)}")
        for f in facts:
            print(f"  • {f}")
        print(f"\nPast sessions: {len(sessions)}")
        for s in sessions:
            print(f"  [{s['last_active'][:10]}] {s['summary']}")
        print("========================\n")