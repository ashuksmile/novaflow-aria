"""SQLite database layer for leads, conversations, appointments, and call summaries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sqlite3
from typing import Any, Dict, List, Optional

from app.config import DATA_DIR, DB_PATH

DEFAULT_TIME_SLOTS = [
    "10:00 AM IST",
    "11:30 AM IST",
    "02:00 PM IST",
    "04:00 PM IST",
    "05:30 PM IST",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            name TEXT DEFAULT '',
            email TEXT DEFAULT '',
            company TEXT DEFAULT '',
            role TEXT DEFAULT '',
            phone TEXT DEFAULT '',
            language TEXT DEFAULT 'en',
            budget TEXT DEFAULT '',
            authority TEXT DEFAULT '',
            need TEXT DEFAULT '',
            timeline TEXT DEFAULT '',
            lead_score INTEGER DEFAULT 10,
            qualification_stage TEXT DEFAULT 'New',
            sentiment_label TEXT DEFAULT 'Neutral',
            sentiment_score REAL DEFAULT 0.0,
            objections_json TEXT DEFAULT '[]',
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            language TEXT DEFAULT 'en',
            sentiment_label TEXT DEFAULT '',
            sentiment_score REAL DEFAULT 0.0,
            rag_sources_json TEXT DEFAULT '[]',
            latency_ms INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            lead_id INTEGER,
            lead_name TEXT NOT NULL,
            lead_email TEXT NOT NULL,
            company TEXT DEFAULT '',
            slot_date TEXT NOT NULL,
            slot_time TEXT NOT NULL,
            status TEXT DEFAULT 'Confirmed',
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS call_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            lead_id INTEGER,
            summary TEXT NOT NULL,
            pain_points_json TEXT DEFAULT '[]',
            objections_json TEXT DEFAULT '[]',
            bant_json TEXT DEFAULT '{}',
            sentiment_trend TEXT DEFAULT 'Neutral',
            recommended_action TEXT DEFAULT '',
            followup_email TEXT DEFAULT '',
            lead_score INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def ensure_lead(session_id: str, language: str = "en") -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM leads WHERE session_id = ?", (session_id,))
    row = cur.fetchone()
    if row is None:
        now = _utc_now()
        cur.execute(
            """
            INSERT INTO leads (session_id, language, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, language, now, now),
        )
        conn.commit()
        cur.execute("SELECT * FROM leads WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
    result = _row_to_lead(row)
    conn.close()
    return result


def update_lead(session_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    ensure_lead(session_id)
    allowed = {
        "name",
        "email",
        "company",
        "role",
        "phone",
        "language",
        "budget",
        "authority",
        "need",
        "timeline",
        "lead_score",
        "qualification_stage",
        "sentiment_label",
        "sentiment_score",
        "objections",
        "notes",
    }
    fields = []
    values = []
    for k, v in updates.items():
        if k not in allowed or v is None:
            continue
        if k == "objections":
            fields.append("objections_json = ?")
            values.append(json.dumps(v))
        else:
            if isinstance(v, str) and not v.strip():
                continue
            fields.append(f"{k} = ?")
            values.append(v)

    if fields:
        fields.append("updated_at = ?")
        values.append(_utc_now())
        values.append(session_id)
        conn = get_connection()
        conn.execute(f"UPDATE leads SET {', '.join(fields)} WHERE session_id = ?", values)
        conn.commit()
        conn.close()

    return ensure_lead(session_id)


def find_returning_lead(email: str = "", name: str = "") -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    row = None
    if email:
        cur.execute(
            "SELECT * FROM leads WHERE LOWER(email) = LOWER(?) ORDER BY updated_at DESC LIMIT 1",
            (email.strip(),),
        )
        row = cur.fetchone()
    elif name:
        cur.execute(
            "SELECT * FROM leads WHERE LOWER(name) = LOWER(?) ORDER BY updated_at DESC LIMIT 1",
            (name.strip(),),
        )
        row = cur.fetchone()
    conn.close()
    return _row_to_lead(row) if row else None


def add_message(
    session_id: str,
    role: str,
    content: str,
    language: str = "en",
    sentiment_label: str = "",
    sentiment_score: float = 0.0,
    rag_sources: Optional[List[dict]] = None,
    latency_ms: int = 0,
) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO messages (
            session_id, role, content, language, sentiment_label,
            sentiment_score, rag_sources_json, latency_ms, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            role,
            content,
            language,
            sentiment_label,
            sentiment_score,
            json.dumps(rag_sources or []),
            latency_ms,
            _utc_now(),
        ),
    )
    conn.commit()
    conn.close()


def get_messages(session_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,)
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["rag_sources"] = json.loads(d.pop("rag_sources_json", "[]") or "[]")
        out.append(d)
    return out


def get_available_slots(date_str: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return available appointment slots for a given YYYY-MM-DD date or the next 3 days."""
    today = datetime.now().date()
    if date_str:
        dates = [date_str]
    else:
        dates = [(today + timedelta(days=i)).isoformat() for i in range(1, 4)]

    conn = get_connection()
    booked_rows = conn.execute(
        "SELECT slot_date, slot_time FROM appointments WHERE status = 'Confirmed'"
    ).fetchall()
    conn.close()
    booked = {(r["slot_date"], r["slot_time"]) for r in booked_rows}

    available = []
    for d in dates:
        free_times = [t for t in DEFAULT_TIME_SLOTS if (d, t) not in booked]
        available.append({"date": d, "available_times": free_times})
    return available


def create_appointment(
    session_id: str,
    lead_name: str,
    lead_email: str,
    slot_date: str,
    slot_time: str,
    company: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    lead = ensure_lead(session_id)
    conn = get_connection()
    cur = conn.cursor()

    existing = cur.execute(
        """
        SELECT id FROM appointments
        WHERE slot_date = ? AND slot_time = ? AND status = 'Confirmed'
        """,
        (slot_date, slot_time),
    ).fetchone()
    if existing:
        conn.close()
        return {"ok": False, "error": f"Slot {slot_date} at {slot_time} is already booked."}

    now = _utc_now()
    cur.execute(
        """
        INSERT INTO appointments (
            session_id, lead_id, lead_name, lead_email, company,
            slot_date, slot_time, status, notes, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 'Confirmed', ?, ?)
        """,
        (session_id, lead["id"], lead_name, lead_email, company, slot_date, slot_time, notes, now),
    )
    appt_id = cur.lastrowid
    conn.commit()
    row = cur.execute("SELECT * FROM appointments WHERE id = ?", (appt_id,)).fetchone()
    conn.close()

    update_lead(
        session_id,
        {
            "name": lead_name,
            "email": lead_email,
            "company": company,
            "qualification_stage": "Demo Booked",
            "lead_score": max(lead.get("lead_score", 10), 85),
        },
    )
    return {"ok": True, "appointment": dict(row)}


def save_call_summary(session_id: str, summary_data: Dict[str, Any]) -> Dict[str, Any]:
    lead = ensure_lead(session_id)
    conn = get_connection()
    now = _utc_now()
    conn.execute(
        """
        INSERT INTO call_summaries (
            session_id, lead_id, summary, pain_points_json, objections_json,
            bant_json, sentiment_trend, recommended_action, followup_email,
            lead_score, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            summary=excluded.summary,
            pain_points_json=excluded.pain_points_json,
            objections_json=excluded.objections_json,
            bant_json=excluded.bant_json,
            sentiment_trend=excluded.sentiment_trend,
            recommended_action=excluded.recommended_action,
            followup_email=excluded.followup_email,
            lead_score=excluded.lead_score,
            created_at=excluded.created_at
        """,
        (
            session_id,
            lead["id"],
            summary_data.get("summary", ""),
            json.dumps(summary_data.get("pain_points", [])),
            json.dumps(summary_data.get("objections", [])),
            json.dumps(summary_data.get("bant", {})),
            summary_data.get("sentiment_trend", "Neutral"),
            summary_data.get("recommended_action", ""),
            summary_data.get("followup_email", ""),
            int(summary_data.get("lead_score", lead.get("lead_score", 0))),
            now,
        ),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM call_summaries WHERE session_id = ?", (session_id,)).fetchone()
    conn.close()
    return _row_to_summary(row)


def get_dashboard_data() -> Dict[str, Any]:
    conn = get_connection()
    # Only sessions where the prospect actually said something count as calls
    # (opening the page creates an empty session).
    leads = [
        _row_to_lead(r)
        for r in conn.execute(
            "SELECT * FROM leads WHERE session_id IN "
            "(SELECT DISTINCT session_id FROM messages WHERE role = 'user') "
            "ORDER BY updated_at DESC"
        ).fetchall()
    ]
    appointments = [
        dict(r) for r in conn.execute("SELECT * FROM appointments ORDER BY created_at DESC").fetchall()
    ]
    summaries = [
        _row_to_summary(r)
        for r in conn.execute("SELECT * FROM call_summaries ORDER BY created_at DESC").fetchall()
    ]
    conn.close()

    total_leads = len(leads)
    qualified = sum(
        1 for l in leads if l["qualification_stage"] in ("Qualified", "Demo Booked") or l["lead_score"] >= 65
    )
    avg_score = round(sum(l["lead_score"] for l in leads) / total_leads, 1) if total_leads else 0.0
    avg_sentiment = (
        round(sum(l["sentiment_score"] for l in leads) / total_leads, 2) if total_leads else 0.0
    )

    return {
        "metrics": {
            "total_calls": total_leads,
            "qualified_leads": qualified,
            "booked_demos": len(appointments),
            "avg_lead_score": avg_score,
            "avg_sentiment": avg_sentiment,
        },
        "leads": leads,
        "appointments": appointments,
        "summaries": summaries,
    }


def _row_to_lead(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["objections"] = json.loads(d.pop("objections_json", "[]") or "[]")
    return d


def _row_to_summary(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["pain_points"] = json.loads(d.pop("pain_points_json", "[]") or "[]")
    d["objections"] = json.loads(d.pop("objections_json", "[]") or "[]")
    d["bant"] = json.loads(d.pop("bant_json", "{}") or "{}")
    return d
