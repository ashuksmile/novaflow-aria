"""Autonomous AI Sales Voice Agent using Groq LLM + Tool Calling + RAG + CRM."""

from __future__ import annotations

from datetime import datetime
import json
import time
from typing import Any, AsyncIterator, Dict, List, Tuple
from groq import AsyncGroq, Groq

from app.config import GROQ_API_KEY, GROQ_CHAT_MODEL
from app.crm import CRMAdapter
from app.rag import LocalRAGEngine

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search NovaFlow AI product documentation for accurate details on features, "
                "pricing plans, SOC2/GDPR/DPDP security, CRM/calendar integrations, ROI case studies, "
                "and objection handling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in English describing what information is needed.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_available_slots",
            "description": "Check open calendar slots for booking a NovaFlow AI product demo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Optional date in YYYY-MM-DD format. Leave empty for next 3 days.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Book a confirmed product demo appointment in the calendar and update the CRM lead. "
                "Only call after confirming the prospect's name, email, date, and time slot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lead_name": {"type": "string", "description": "Prospect full name"},
                    "lead_email": {"type": "string", "description": "Prospect email address"},
                    "company": {"type": "string", "description": "Prospect company name"},
                    "slot_date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "slot_time": {"type": "string", "description": "Exact time slot string, e.g. '11:30 AM IST'"},
                    "notes": {"type": "string", "description": "Short summary of prospect needs"},
                },
                "required": ["lead_name", "lead_email", "slot_date", "slot_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_returning_customer",
            "description": "Look up an existing lead in the CRM by email or name to recall past context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "Prospect email"},
                    "name": {"type": "string", "description": "Prospect name"},
                },
            },
        },
    },
]


def _build_system_prompt(language: str, lead: Dict[str, Any]) -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    lang_rule = (
        "The user is speaking Hindi. Respond in simple, conversational Hindi written ONLY in "
        "Devanagari script. Write English product words in Devanagari too "
        "(नोवाफ़्लो, सीआरएम, डेमो, प्लान) because a Hindi voice reads your reply aloud."
        if language == "hi"
        else "Respond naturally in clear, concise spoken English."
    )

    return f"""You are Aria, an autonomous AI Sales Representative for NovaFlow AI (a B2B SaaS RevOps & AI Sales Automation platform).
Today's date is {today}.

VOICE CALL RULES (you are on a live phone-style voice call; everything you write is spoken aloud by TTS):
- Keep every reply short and natural for speech: 1 to 3 short sentences.
- NEVER use markdown, bullet points, lists, emojis, URLs, or symbols like $ % / &. Say "799 dollars per month", "15 percent".
- The caller's words come from speech recognition and may contain small errors; infer the intended meaning.
- If the caller interrupted you, do not repeat everything; answer what they just said.
- Ask only ONE clear follow-up question at a time.
- {lang_rule}

SALES PLAYBOOK:
1. Be warm, consultative, and genuinely helpful—never pushy.
2. Whenever the prospect asks about pricing, features, security, integrations, or ROI, ALWAYS call `search_knowledge_base` first so your facts are 100% accurate.
3. Naturally uncover BANT during the conversation: Need (pain point), Budget/team size, Authority (role), and Timeline.
4. Once the prospect shows interest, offer to book a 20-minute live demo. Call `check_available_slots`, ask for their name, email, and preferred slot, then call `book_appointment`.
5. If the prospect mentions their name or email early, you may call `lookup_returning_customer` to personalize the conversation.

CURRENT CRM CONTEXT FOR THIS CALL:
Name: {lead.get('name') or 'Unknown'} | Company: {lead.get('company') or 'Unknown'} | Stage: {lead.get('qualification_stage', 'New')} | Score: {lead.get('lead_score', 10)}/100
Known Need: {lead.get('need') or 'Not yet discovered'} | Budget: {lead.get('budget') or 'Unknown'} | Timeline: {lead.get('timeline') or 'Unknown'}
"""


class SalesVoiceAgent:
    """Orchestrates multi-turn conversation, RAG retrieval, and CRM tool execution."""

    def __init__(self, rag: LocalRAGEngine, crm: CRMAdapter) -> None:
        self.client = Groq(api_key=GROQ_API_KEY)
        self.async_client = AsyncGroq(api_key=GROQ_API_KEY)
        self.rag = rag
        self.crm = crm

    async def respond_stream(
        self,
        session_id: str,
        user_text: str,
        history: List[Dict[str, str]],
        language: str,
        lead: Dict[str, Any],
        rag_sources: List[dict],
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Streaming agent turn for the voice pipeline.

        Yields ("text", delta) as tokens arrive so TTS can start on the first sentence,
        and ("tool", {"tool", "args", "result"}) whenever a tool runs.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _build_system_prompt(language, lead)}
        ]
        for m in history[-12:]:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_text})

        for step in range(4):
            use_tools = step < 3
            kwargs: Dict[str, Any] = dict(
                model=GROQ_CHAT_MODEL,
                messages=messages,
                temperature=0.35,
                max_tokens=400,
                stream=True,
            )
            if "gpt-oss" in GROQ_CHAT_MODEL:
                kwargs["reasoning_effort"] = "low"
            if use_tools:
                kwargs.update(tools=TOOLS, tool_choice="auto")
            stream = await self.async_client.chat.completions.create(**kwargs)

            calls: Dict[int, Dict[str, str]] = {}
            content = ""
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    content += delta.content
                    yield "text", delta.content
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments

            if not calls:
                return

            ordered = [calls[i] for i in sorted(calls)]
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in ordered
                    ],
                }
            )
            for c in ordered:
                try:
                    args = json.loads(c["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                yield "tool_start", {"tool": c["name"], "args": args}
                result = self._execute_tool(session_id, c["name"], args, rag_sources)
                yield "tool", {"tool": c["name"], "args": args, "result": result}
                messages.append(
                    {"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                     "content": json.dumps(result, default=str)}
                )

    def respond(
        self,
        session_id: str,
        user_text: str,
        history: List[Dict[str, str]],
        language: str,
        lead: Dict[str, Any],
    ) -> Tuple[str, List[dict], List[dict], int]:
        """Run the agent turn with tool calling.

        Returns:
            (assistant_reply, rag_sources, tool_events, llm_latency_ms)
        """
        t0 = time.perf_counter()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _build_system_prompt(language, lead)}
        ]
        # Keep rolling window of last 12 turns for fast, focused context
        for m in history[-12:]:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_text})

        rag_sources: List[dict] = []
        tool_events: List[dict] = []

        for _ in range(3):
            resp = self.client.chat.completions.create(
                model=GROQ_CHAT_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.35,
                max_tokens=300,
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                reply = (msg.content or "").strip()
                latency_ms = int((time.perf_counter() - t0) * 1000)
                return reply, rag_sources, tool_events, latency_ms

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                result = self._execute_tool(session_id, fn_name, args, rag_sources)
                tool_events.append({"tool": fn_name, "args": args, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fn_name,
                        "content": json.dumps(result),
                    }
                )

        # Final synthesis call if tool loop reached max iterations
        final_resp = self.client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=messages,
            temperature=0.35,
            max_tokens=300,
        )
        reply = (final_resp.choices[0].message.content or "").strip()
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return reply, rag_sources, tool_events, latency_ms

    def _execute_tool(
        self,
        session_id: str,
        fn_name: str,
        args: Dict[str, Any],
        rag_sources: List[dict],
    ) -> Any:
        if fn_name == "search_knowledge_base":
            query = str(args.get("query", ""))
            hits = self.rag.search(query, top_k=3)
            rag_sources.extend(hits)
            return {"matches": hits}

        if fn_name == "check_available_slots":
            slots = self.crm.check_slots(date_str=args.get("date"))
            return {"slots": slots}

        if fn_name == "book_appointment":
            return self.crm.book_demo(
                session_id=session_id,
                lead_name=str(args.get("lead_name", "Prospect")),
                lead_email=str(args.get("lead_email", "")),
                slot_date=str(args.get("slot_date", "")),
                slot_time=str(args.get("slot_time", "")),
                company=str(args.get("company", "")),
                notes=str(args.get("notes", "")),
            )

        if fn_name == "lookup_returning_customer":
            found = self.crm.lookup_lead(
                email=str(args.get("email", "")),
                name=str(args.get("name", "")),
            )
            return {"found": bool(found), "lead": found}

        return {"error": f"Unknown tool: {fn_name}"}
