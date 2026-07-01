"""Provenance Guard — Milestones 3-5.

Flask service that accepts a text submission, runs two independent detection
signals (an LLM classifier via Groq + stylometric heuristics), combines them into
one calibrated confidence score, assigns a transparency label, writes a structured
entry to a SQLite audit log, and lets creators appeal a result.

Endpoints:
    POST /submit    -> classify a submission (both signals + confidence + label)
    POST /appeal    -> file an appeal against a prior classification
    GET  /appeals   -> reviewer queue of appealed submissions
    GET  /log       -> most recent audit-log entries
    GET  /health    -> liveness check
"""

import os
import uuid
import json
import sqlite3
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from signals import groq_signal, stylometric_signal

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "audit_log.db")
MAX_TEXT_CHARS = 20_000

# Signal combination weights (see planning.md §3).
W_LLM = 0.6
W_STYLO = 0.4
W_STYLO_UNRELIABLE = 0.15  # down-weight stylometry when text is too short

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


# --------------------------------------------------------------------------- #
# Audit log (SQLite)
# --------------------------------------------------------------------------- #
def init_db():
    """Create the audit-log table if it does not exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id        TEXT    NOT NULL,
                creator_id        TEXT,
                timestamp         TEXT    NOT NULL,
                event             TEXT    NOT NULL,
                attribution       TEXT,
                confidence        REAL,
                llm_score         REAL,
                stylo_score       REAL,
                status            TEXT,
                appeal_reasoning  TEXT,
                detail            TEXT
            )
            """
        )
        conn.commit()


def write_log(entry: dict):
    """Append one structured entry to the audit log."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO audit_log
                (content_id, creator_id, timestamp, event, attribution,
                 confidence, llm_score, stylo_score, status, appeal_reasoning, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.get("content_id"),
                entry.get("creator_id"),
                entry.get("timestamp"),
                entry.get("event", "classified"),
                entry.get("attribution"),
                entry.get("confidence"),
                entry.get("llm_score"),
                entry.get("stylo_score"),
                entry.get("status"),
                entry.get("appeal_reasoning"),
                json.dumps(entry.get("detail")) if entry.get("detail") else None,
            ),
        )
        conn.commit()


def get_log(limit: int = 50) -> list:
    """Return the most recent audit-log entries, newest first."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    entries = []
    for row in rows:
        entry = dict(row)
        if entry.get("detail"):
            entry["detail"] = json.loads(entry["detail"])
        entries.append(entry)
    return entries


def latest_classification(content_id: str):
    """Return the most recent classification row for a content_id, or None."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT * FROM audit_log
            WHERE content_id = ? AND event = 'classified'
            ORDER BY id DESC LIMIT 1
            """,
            (content_id,),
        ).fetchone()
    return dict(row) if row else None


def now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


# --------------------------------------------------------------------------- #
# Confidence scoring (Milestone 4) — see planning.md §3
# --------------------------------------------------------------------------- #
def combine_signals(signal1: dict, signal2: dict) -> dict:
    """Combine the two signals into one calibrated confidence score.

    Returns {confidence, certainty, attribution} where `confidence` is the
    combined AI-likelihood in [0,1].
    """
    llm_ok = signal1.get("ok", False)
    stylo_ok = signal2.get("ok", False)
    llm = signal1["ai_likelihood"]
    stylo = signal2["ai_likelihood"]

    if not llm_ok and not stylo_ok:
        raw = 0.5
    elif not llm_ok:
        raw = stylo  # LLM unavailable -> stylometry alone
    else:
        w_stylo = W_STYLO if stylo_ok else W_STYLO_UNRELIABLE
        raw = (W_LLM * llm + w_stylo * stylo) / (W_LLM + w_stylo)

    # Short-text penalty: thin evidence gets pulled toward 0.5 (uncertain).
    feats = signal2.get("features", {})
    if feats.get("n_sentences", 0) < 3 or feats.get("n_words", 0) < 40:
        raw = 0.5 + (raw - 0.5) * 0.6

    confidence = round(max(0.0, min(1.0, raw)), 4)
    certainty = round(2 * abs(confidence - 0.5), 4)

    if confidence >= 0.65:
        attribution = "likely_ai"
    elif confidence <= 0.35:
        attribution = "likely_human"
    else:
        attribution = "uncertain"

    return {
        "confidence": confidence,
        "certainty": certainty,
        "attribution": attribution,
    }


# --------------------------------------------------------------------------- #
# Transparency labels (Milestone 5) — see planning.md §4
# --------------------------------------------------------------------------- #
def build_label(attribution: str, confidence: float) -> str:
    """Map an attribution + confidence to one of three plain-English labels."""
    pct = round(confidence * 100)
    if attribution == "likely_ai":
        return (
            f"⚑ Likely AI-generated. Our analysis estimates a {pct}% chance "
            f"this text was produced by AI, based on two independent signals. If "
            f"you believe this is wrong, you can appeal using your content ID."
        )
    if attribution == "likely_human":
        return (
            f"✓ Likely human-written. Our analysis found no strong signs of AI "
            f"generation ({pct}% AI-likelihood). This is an estimate, not a "
            f"guarantee."
        )
    return (
        f"? Inconclusive. Our signals disagree or are weak ({pct}% AI-likelihood, "
        f"low certainty). We are not confident either way; treat this result with "
        f"caution and appeal if it affects you."
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/submit")
@limiter.limit("10 per minute;100 per day")
def submit():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    creator_id = data.get("creator_id")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be non-empty."}), 400
    if not creator_id:
        return jsonify({"error": "Field 'creator_id' is required."}), 400
    if len(text) > MAX_TEXT_CHARS:
        return jsonify({"error": f"'text' exceeds {MAX_TEXT_CHARS} characters."}), 413

    content_id = str(uuid.uuid4())

    # --- Two independent detection signals ---
    signal1 = groq_signal(text)               # semantic (Groq LLM)
    signal2 = stylometric_signal(text)        # structural (pure Python)

    # --- Confidence scoring + transparency label ---
    scored = combine_signals(signal1, signal2)
    attribution = scored["attribution"]
    confidence = scored["confidence"]
    label = build_label(attribution, confidence)

    timestamp = now_iso()
    write_log(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "timestamp": timestamp,
            "event": "classified",
            "attribution": attribution,
            "confidence": confidence,
            "llm_score": signal1["ai_likelihood"],
            "stylo_score": signal2["ai_likelihood"],
            "status": "classified",
            "detail": {"signal1": signal1, "signal2": signal2, "scoring": scored},
        }
    )

    return jsonify(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "timestamp": timestamp,
            "attribution": attribution,
            "confidence": confidence,
            "certainty": scored["certainty"],
            "label": label,
            "signals": {"signal1_llm": signal1, "signal2_stylometric": signal2},
            "status": "classified",
        }
    )


@app.post("/appeal")
@limiter.limit("20 per minute")
def appeal():
    data = request.get_json(silent=True) or {}
    content_id = data.get("content_id")
    creator_reasoning = data.get("creator_reasoning")

    if not content_id:
        return jsonify({"error": "Field 'content_id' is required."}), 400
    if not isinstance(creator_reasoning, str) or not creator_reasoning.strip():
        return jsonify({"error": "Field 'creator_reasoning' is required."}), 400

    original = latest_classification(content_id)
    if original is None:
        return jsonify({"error": f"No classification found for content_id {content_id}."}), 404

    timestamp = now_iso()
    # Log the appeal alongside the original decision; the original row is preserved.
    write_log(
        {
            "content_id": content_id,
            "creator_id": original.get("creator_id"),
            "timestamp": timestamp,
            "event": "appeal",
            "attribution": original.get("attribution"),
            "confidence": original.get("confidence"),
            "llm_score": original.get("llm_score"),
            "stylo_score": original.get("stylo_score"),
            "status": "under_review",
            "appeal_reasoning": creator_reasoning,
        }
    )

    return jsonify(
        {
            "content_id": content_id,
            "status": "under_review",
            "message": "Appeal received. Your submission is now queued for human review.",
            "original_attribution": original.get("attribution"),
            "original_confidence": original.get("confidence"),
            "timestamp": timestamp,
        }
    )


@app.get("/appeals")
def appeals():
    """Reviewer queue: every appeal event, newest first."""
    queue = [e for e in get_log(limit=200) if e.get("event") == "appeal"]
    return jsonify({"appeals": queue, "count": len(queue)})


@app.get("/log")
def log():
    return jsonify({"entries": get_log()})


init_db()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
