"""Real-time sentiment analysis, BANT lead qualification, and post-call business insights."""

from __future__ import annotations

import json
from typing import Any, Dict, List
from groq import Groq

from app.config import GROQ_API_KEY, GROQ_CHAT_MODEL, GROQ_FAST_MODEL


class SalesInsightsEngine:
    """Uses Groq JSON mode for fast per-turn intelligence and rich end-of-call summaries."""

    def __init__(self) -> None:
        self.client = Groq(api_key=GROQ_API_KEY)

    def analyze_turn(
        self,
        history: List[Dict[str, str]],
        latest_user_text: str,
        current_lead: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract BANT signals, lead profile details, sentiment, and objections in JSON."""
        recent_turns = history[-8:]
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent_turns)

        prompt = f"""You are a B2B sales intelligence analyzer.
Analyze the conversation and the latest prospect message.
Return ONLY valid JSON with these exact keys:
{{
  "name": "string or empty if unknown",
  "email": "string or empty if unknown",
  "company": "string or empty if unknown",
  "role": "string or empty if unknown",
  "budget": "short summary of budget signal or empty",
  "authority": "short summary of decision-making authority or empty",
  "need": "short summary of pain point / need or empty",
  "timeline": "short summary of buying timeline or empty",
  "lead_score": integer 0 to 100,
  "qualification_stage": "New" | "Discovery" | "Qualified" | "Demo Booked" | "Disqualified",
  "sentiment_label": "Positive" | "Neutral" | "Hesitant" | "Frustrated",
  "sentiment_score": float between -1.0 and 1.0,
  "objections": ["list of short objection tags if any"]
}}

Current stored lead profile:
{json.dumps({k: current_lead.get(k) for k in ("name", "email", "company", "role", "budget", "authority", "need", "timeline", "lead_score", "qualification_stage")})}

Recent conversation:
{transcript}

Latest prospect message:
{latest_user_text}

Preserve previously known fields unless the prospect provides newer information."""

        try:
            resp = self.client.chat.completions.create(
                model=GROQ_FAST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=400,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            # Keep Demo Booked stage if already booked
            if current_lead.get("qualification_stage") == "Demo Booked":
                data["qualification_stage"] = "Demo Booked"
                data["lead_score"] = max(int(data.get("lead_score", 85)), 85)
            return data
        except Exception:
            return {
                "sentiment_label": current_lead.get("sentiment_label", "Neutral"),
                "sentiment_score": current_lead.get("sentiment_score", 0.0),
                "lead_score": current_lead.get("lead_score", 15),
                "qualification_stage": current_lead.get("qualification_stage", "Discovery"),
            }

    def generate_call_summary(
        self,
        messages: List[Dict[str, Any]],
        lead: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate structured executive call summary, business insights, and follow-up email."""
        transcript = "\n".join(
            f"{m['role'].upper()} ({m.get('language', 'en')}): {m['content']}" for m in messages
        )
        if not transcript.strip():
            transcript = "(No conversation recorded yet.)"

        prompt = f"""You are a VP of Sales reviewing a completed AI sales voice call for NovaFlow AI.
Produce a concise, actionable post-call intelligence report in valid JSON with these exact keys:
{{
  "summary": "2-3 sentence executive summary of the call outcome",
  "pain_points": ["list of 1-4 concrete buyer pain points"],
  "objections": ["list of buyer objections raised, or empty list"],
  "bant": {{
    "budget": "status/detail",
    "authority": "status/detail",
    "need": "status/detail",
    "timeline": "status/detail"
  }},
  "sentiment_trend": "e.g., Neutral -> Positive",
  "lead_score": integer 0 to 100,
  "recommended_action": "clear next step for the human Account Executive",
  "followup_email": "short, personalized follow-up email (Subject + body) ready to send"
}}

Lead CRM state:
{json.dumps(lead)}

Full Call Transcript:
{transcript}"""

        resp = self.client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=800,
        )
        return json.loads(resp.choices[0].message.content or "{}")
