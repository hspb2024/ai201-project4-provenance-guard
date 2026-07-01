"""Provenance Guard — Milestone 3.

Flask service that accepts a text submission, runs the first detection signal
(an LLM classifier via Groq), assigns a placeholder confidence + label, writes a
structured entry to a SQLite audit log, and exposes the log.

Endpoints:
    POST /submit   -> classify a submission (Signal 1 wired; confidence/label are
                      placeholders until Milestone 4/5)
    GET  /log      -> most recent audit-log entries
    GET  /health   -> liveness check
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

from signals import groq_signal

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "audit_log.db")
MAX_TEXT_CHARS = 20_000

app = Flask(__name__)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["60 per minute"],
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
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id  TEXT    NOT NULL,
                creator_id  TEXT,
                timestamp   TEXT    NOT NULL,
                event       TEXT    NOT NULL,
                attribution TEXT,
                confidence  REAL,
                llm_score   REAL,
                status      TEXT,
                detail      TEXT
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
                 confidence, llm_score, status, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.get("content_id"),
                entry.get("creator_id"),
                entry.get("timestamp"),
                entry.get("event", "classified"),
                entry.get("attribution"),
                entry.get("confidence"),
                entry.get("llm_score"),
                entry.get("status"),
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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


# --------------------------------------------------------------------------- #
# Placeholder scoring / labeling (real logic arrives in M4/M5)
# --------------------------------------------------------------------------- #
def placeholder_attribution(llm_score: float) -> str:
    if llm_score >= 0.65:
        return "likely_ai"
    if llm_score <= 0.35:
        return "likely_human"
    return "uncertain"


def placeholder_label(attribution: str, llm_score: float) -> str:
    pct = round(llm_score * 100)
    return (
        f"[placeholder — Signal 1 only] attribution={attribution}, "
        f"{pct}% AI-likelihood. Confidence scoring and final labels land in "
        f"Milestone 4/5."
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/submit")
@limiter.limit("20 per minute")
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

    # --- Signal 1: LLM classifier (Groq) ---
    signal1 = groq_signal(text)
    llm_score = signal1["ai_likelihood"]

    # --- Placeholder scoring + label (M4/M5 will replace these) ---
    attribution = placeholder_attribution(llm_score)
    confidence = round(2 * abs(llm_score - 0.5), 4)  # placeholder certainty
    label = placeholder_label(attribution, llm_score)

    timestamp = now_iso()
    write_log(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "timestamp": timestamp,
            "event": "classified",
            "attribution": attribution,
            "confidence": confidence,
            "llm_score": llm_score,
            "status": "classified",
            "detail": {"signal1": signal1},
        }
    )

    return jsonify(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "timestamp": timestamp,
            "attribution": attribution,
            "confidence": confidence,
            "label": label,
            "signals": {"signal1_llm": signal1},
            "status": "classified",
        }
    )


@app.get("/log")
def log():
    return jsonify({"entries": get_log()})


init_db()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
