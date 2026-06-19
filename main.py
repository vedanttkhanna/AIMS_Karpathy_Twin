import os
import sys
import uuid
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

from agent.teaching import generate_response
from agent.advisory import think
from agent.guard import classify_query, get_guard_response
from agent.feedback import analyze_feedback, AdaptationState
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from ingest.embedder import build_index

console = Console()

BANNER = """
╔═══════════════════════════════════════════════════╗
║         Andrej Karpathy — Digital Twin            ║
║   Type /think <problem> for advisory mode         ║
║   Type /memory to see what I remember             ║
║   Type /reward to see conversation reward signal  ║
║   Type /clear to reset conversation               ║
║   Type /quit to exit                              ║
╚═══════════════════════════════════════════════════╝
"""


def print_banner():
    console.print(BANNER, style="bold cyan")


def print_response(text: str, mode: str = "teaching"):
    color = "green" if mode == "teaching" else "yellow"
    label = "Karpathy" if mode == "teaching" else "Karpathy [thinking]"
    console.print(Panel(
        Markdown(text),
        title=f"[bold {color}]{label}[/]",
        border_style=color,
        padding=(1, 2)
    ))


def print_feedback_signal(sentiment: str, reward: float):
    """Show subtle adaptation signal so user knows the system is learning."""
    color = "green" if reward >= 0 else "red"
    console.print(
        f"[dim {color}][ adaptation signal: {sentiment} | reward: {reward:+.1f} ][/dim {color}]"
    )


def summarize_session(short_term: ShortTermMemory) -> str:
    history = short_term.get()
    if not history:
        return ""
    topics = [m["content"][:60] for m in history if m["role"] == "user"]
    return f"Discussed: {', '.join(topics[:3])}"


def main():
    print_banner()

    console.print("[dim]Checking RAG index...[/dim]")
    build_index()

    short_term = ShortTermMemory()
    long_term = LongTermMemory()
    adaptation_state = AdaptationState()    # NEW
    session_id = str(uuid.uuid4())[:8]
    last_response = ""                      # NEW — track last response for feedback analysis

    console.print(f"[dim]Session {session_id} started at {datetime.now().strftime('%H:%M')}[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        # commands
        if user_input.lower() == "/quit":
            summary = summarize_session(short_term)
            if summary:
                long_term.save_session_summary(session_id, summary)
            console.print("[dim]Session saved. Goodbye.[/dim]")
            break

        elif user_input.lower() == "/clear":
            short_term.clear()
            adaptation_state = AdaptationState()   # reset adaptation on clear
            console.print("[dim]Conversation and adaptation state cleared.[/dim]")
            continue

        elif user_input.lower() == "/memory":
            long_term.show()
            continue

        elif user_input.lower() == "/reward":
            reward = adaptation_state.get_reward()
            depth = adaptation_state.depth_level
            color = "green" if reward >= 0 else "red"
            console.print(f"\n[bold]Conversation Reward Signal[/bold]")
            console.print(f"  Reward score:     [{color}]{reward:+.1f}[/{color}]")
            console.print(f"  Depth level:      {depth} (-2 simple ←→ +2 technical)")
            console.print(f"  Confusion count:  {adaptation_state.confusion_count}")
            console.print(f"  Satisfaction:     {adaptation_state.satisfaction_count}")
            if adaptation_state.style_notes:
                console.print(f"  Recent signals:   {'; '.join(adaptation_state.style_notes[-3:])}")
            print()
            continue

        elif user_input.lower().startswith("/think "):
            problem = user_input[7:].strip()
            if not problem:
                console.print("[red]Usage: /think <your problem>[/red]")
                continue
            console.print("[dim yellow]Thinking...[/dim yellow]")
            response = think(problem, short_term)
            print_response(response, mode="advisory")
            short_term.add("user", f"[think] {problem}")
            short_term.add("assistant", response)
            last_response = response
            continue

        # ── RL feedback loop ──────────────────────────────────────────
        # if there's a previous response, analyze this message as potential feedback
        if last_response:
            feedback = analyze_feedback(user_input, last_response)
            if feedback.get("is_feedback") and feedback.get("confidence", 0) > 0.5:
                adaptation_state.update(
                    feedback["sentiment"],
                    feedback.get("adjustment", "none")
                )
                print_feedback_signal(
                    feedback["sentiment"],
                    adaptation_state.get_reward()
                )
        # ─────────────────────────────────────────────────────────────

        # guard layer
        classification = classify_query(user_input)
        if classification in ("attack", "offtopic"):
            response = get_guard_response(classification)
            print_response(response)
            short_term.add("user", user_input)
            short_term.add("assistant", response)
            last_response = response
            continue

        # generate response with adaptation state
        response = generate_response(
            user_input, short_term, long_term, session_id,
            adaptation_state=adaptation_state    # NEW
        )
        print_response(response, mode="teaching")
        short_term.add("user", user_input)
        short_term.add("assistant", response)
        last_response = response                 # NEW


if __name__ == "__main__":
    main()