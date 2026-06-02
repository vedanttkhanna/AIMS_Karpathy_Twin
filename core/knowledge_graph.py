import json
import os
from pathlib import Path
from config import KNOWLEDGE_BASE_DIR

# ─── Graph Structure ──────────────────────────────────────────────────────────
# Built in memory at startup from knowledge_base JSONs
# Nodes = concepts/projects
# Edges = "Karpathy connects these"

class KarpathyKnowledgeGraph:
    def __init__(self):
        self.nodes = {}   # name → {type, description, source_project}
        self.edges = {}   # name → [related_names]
        self._build()

    def _build(self):
        kb_path = Path(KNOWLEDGE_BASE_DIR)
        
        # load all project JSONs
        for json_file in kb_path.glob("*.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                
                if "project" in data:
                    project_name = data["project"]
                    
                    # add project as node
                    self.add_node(project_name, {
                        "type": "project",
                        "description": data.get("description", ""),
                        "lessons": data.get("lessons", []),
                        "key_decisions": data.get("key_decisions", []),
                        "github": data.get("github", "")
                    })
                    
                    # add connected concepts as nodes + edges
                    for concept in data.get("connected_concepts", []):
                        self.add_node(concept, {"type": "concept"})
                        self.add_edge(project_name, concept)
                        self.add_edge(concept, project_name)

                elif "core_principles" in data:
                    for p in data["core_principles"]:
                        principle_name = p["principle"]
                        self.add_node(principle_name, {
                            "type": "principle",
                            "explanation": p.get("explanation", ""),
                            "seen_in": p.get("seen_in", [])
                        })
                        # connect principle to projects it appears in
                        for project in p.get("seen_in", []):
                            self.add_edge(principle_name, project)
                            self.add_edge(project, principle_name)

            except Exception as e:
                print(f"[kg] failed to load {json_file}: {e}")

        print(f"[kg] built graph: {len(self.nodes)} nodes, {sum(len(v) for v in self.edges.values())} edges")

    def add_node(self, name: str, data: dict):
        if name not in self.nodes:
            self.nodes[name] = data

    def add_edge(self, from_node: str, to_node: str):
        if from_node not in self.edges:
            self.edges[from_node] = []
        if to_node not in self.edges[from_node]:
            self.edges[from_node].append(to_node)

    def get_related(self, query: str, depth: int = 2) -> dict:
        """
        Given a query, find relevant nodes by keyword match
        then traverse edges to depth N.
        Returns dict of relevant projects + principles + concepts.
        """
        query_lower = query.lower()
        
        # find seed nodes that match query keywords
        seeds = []
        for node_name in self.nodes:
            if any(word in node_name.lower() for word in query_lower.split()):
                seeds.append(node_name)
            # also check node descriptions
            node_data = self.nodes[node_name]
            desc = str(node_data.get("description", "")).lower()
            explanation = str(node_data.get("explanation", "")).lower()
            if any(word in desc + explanation for word in query_lower.split() if len(word) > 4):
                if node_name not in seeds:
                    seeds.append(node_name)

        # traverse to collect related nodes
        visited = set(seeds)
        frontier = list(seeds)
        for _ in range(depth):
            next_frontier = []
            for node in frontier:
                for neighbor in self.edges.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier

        # organize results by type
        result = {
            "projects": [],
            "principles": [],
            "concepts": []
        }
        for node_name in visited:
            node_data = self.nodes.get(node_name, {})
            node_type = node_data.get("type", "concept")
            entry = {"name": node_name, **node_data}
            if node_type == "project":
                result["projects"].append(entry)
            elif node_type == "principle":
                result["principles"].append(entry)
            else:
                result["concepts"].append(entry)

        return result

    def format_for_prompt(self, query: str) -> str:
        """Format relevant graph context for injection into prompts."""
        related = self.get_related(query)
        parts = []

        if related["projects"]:
            parts.append("## Relevant projects Karpathy has built:")
            for p in related["projects"]:
                parts.append(f"\n### {p['name']}")
                parts.append(p.get("description", ""))
                decisions = p.get("key_decisions", [])
                if decisions:
                    parts.append("Key decisions:")
                    for d in decisions[:3]:
                        parts.append(f"  - {d['decision']}: {d['reasoning']}")
                lessons = p.get("lessons", [])
                if lessons:
                    parts.append("Lessons:")
                    for l in lessons[:2]:
                        parts.append(f"  - {l}")

        if related["principles"]:
            parts.append("\n## Relevant principles:")
            for p in related["principles"]:
                parts.append(f"  - {p['name']}: {p.get('explanation', '')}")

        return "\n".join(parts) if parts else ""


# singleton
_kg = None

def get_kg() -> KarpathyKnowledgeGraph:
    global _kg
    if _kg is None:
        _kg = KarpathyKnowledgeGraph()
    return _kg