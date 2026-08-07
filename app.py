import sys
import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session
from core.key_manager import key_manager, QuotaExceededError, InvalidApiKeyError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.secret_key = os.urandom(24)

# store per-session state in memory
_sessions = {}


def get_session_state(session_id):
    if session_id not in _sessions:
        from memory.short_term import ShortTermMemory
        from memory.long_term import LongTermMemory
        from agent.feedback import AdaptationState
        _sessions[session_id] = {
            "short_term": ShortTermMemory(),
            "long_term": LongTermMemory(),
            "adaptation": AdaptationState(),
            "last_response": "",
            "created": datetime.now().isoformat(),
        }
    return _sessions[session_id]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/init", methods=["POST"])
def init_session():
    """Called when user enters Groq API key(s). Sets up key_manager & RAG index."""
    data = request.json
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "Groq API key required"}), 400

    key_manager.set_keys(api_key)

    try:
        from ingest.embedder import build_index
        build_index()

        session_id = str(uuid.uuid4())[:8]
        session["session_id"] = session_id
        return jsonify({"session_id": session_id, "status": "ready"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset-keys", methods=["POST"])
def reset_keys():
    """Completely clears registered keys and session states for a fresh run."""
    key_manager.keys = []
    key_manager.current_idx = 0
    if "GROQ_API_KEY" in os.environ:
        del os.environ["GROQ_API_KEY"]
    _sessions.clear()
    session.clear()
    return jsonify({"status": "cleared"})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    mode = data.get("mode", "teach")
    session_id = session.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "empty message"}), 400

    state = get_session_state(session_id)
    short_term = state["short_term"]
    long_term = state["long_term"]
    adaptation = state["adaptation"]
    last_response = state["last_response"]

    try:
        feedback_signal = None

        if last_response and mode == "teach":
            from agent.feedback import analyze_feedback
            try:
                feedback = analyze_feedback(user_message, last_response)
                if feedback.get("is_feedback") and feedback.get("confidence", 0) > 0.5:
                    adaptation.update(
                        feedback["sentiment"],
                        feedback.get("adjustment", "")
                    )
                    feedback_signal = {
                        "sentiment": feedback["sentiment"],
                        "reward": adaptation.get_reward(),
                        "depth_level": adaptation.depth_level,
                    }
            except Exception as e:
                print(f"[feedback] failed: {e}")

        if mode == "think":
            from agent.advisory import think
            response = think(user_message, short_term)
            short_term.add("user", f"[think] {user_message}")
        else:
            from agent.guard import classify_query, get_guard_response
            classification = classify_query(user_message)
            if classification in ("attack", "offtopic"):
                response = get_guard_response(classification)
            else:
                from agent.teaching import generate_response
                response = generate_response(
                    user_message, short_term, long_term,
                    session_id, adaptation_state=adaptation
                )
            short_term.add("user", user_message)

        short_term.add("assistant", response)
        state["last_response"] = response

        return jsonify({
            "response": response,
            "feedback_signal": feedback_signal,
            "reward": adaptation.get_reward(),
            "depth_level": adaptation.depth_level,
            "confusion_count": adaptation.confusion_count,
            "satisfaction_count": adaptation.satisfaction_count,
        })

    except (QuotaExceededError, InvalidApiKeyError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/memory", methods=["GET"])
def memory():
    session_id = session.get("session_id", "default")
    state = get_session_state(session_id)
    long_term = state["long_term"]
    facts = long_term.get_all_facts()
    sessions = long_term.get_recent_sessions()
    return jsonify({"facts": facts, "sessions": sessions})


@app.route("/api/reward", methods=["GET"])
def reward():
    session_id = session.get("session_id", "default")
    state = get_session_state(session_id)
    a = state["adaptation"]
    return jsonify({
        "reward": a.get_reward(),
        "depth_level": a.depth_level,
        "confusion_count": a.confusion_count,
        "satisfaction_count": a.satisfaction_count,
        "style_notes": a.style_notes[-5:],
    })


@app.route("/api/clear", methods=["POST"])
def clear():
    session_id = session.get("session_id", "default")
    if session_id in _sessions:
        from memory.short_term import ShortTermMemory
        from agent.feedback import AdaptationState
        _sessions[session_id]["short_term"] = ShortTermMemory()
        _sessions[session_id]["adaptation"] = AdaptationState()
        _sessions[session_id]["last_response"] = ""
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)