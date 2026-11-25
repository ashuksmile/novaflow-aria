"""Pluggable CRM Adapter abstraction (SQLite default + optional webhook sync)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import httpx

from app import database as db
from app.config import CRM_WEBHOOK_URL


class CRMAdapter(ABC):
    """Interface for syncing leads, appointments, and call summaries to any CRM."""

    @abstractmethod
    def upsert_lead(self, session_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def lookup_lead(self, email: str = "", name: str = "") -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def check_slots(self, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def book_demo(
        self,
        session_id: str,
        lead_name: str,
        lead_email: str,
        slot_date: str,
        slot_time: str,
        company: str = "",
        notes: str = "",
    ) -> Dict[str, Any]:
        ...


class SQLiteCRMAdapter(CRMAdapter):
    """Local SQLite CRM with optional fire-and-forget webhook push (HubSpot/Salesforce/Zapier)."""

    def __init__(self, webhook_url: str = CRM_WEBHOOK_URL):
        self.webhook_url = webhook_url.strip()

    def _emit_webhook(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.webhook_url:
            return
        try:
            httpx.post(
                self.webhook_url,
                json={"event": event, "data": payload},
                timeout=2.5,
            )
        except Exception:
            # Never block the real-time voice call on an external webhook failure.
            pass

    def upsert_lead(self, session_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        lead = db.update_lead(session_id, updates)
        self._emit_webhook("lead.updated", lead)
        return lead

    def lookup_lead(self, email: str = "", name: str = "") -> Optional[Dict[str, Any]]:
        return db.find_returning_lead(email=email, name=name)

    def check_slots(self, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
        return db.get_available_slots(date_str=date_str)

    def book_demo(
        self,
        session_id: str,
        lead_name: str,
        lead_email: str,
        slot_date: str,
        slot_time: str,
        company: str = "",
        notes: str = "",
    ) -> Dict[str, Any]:
        res = db.create_appointment(
            session_id=session_id,
            lead_name=lead_name,
            lead_email=lead_email,
            slot_date=slot_date,
            slot_time=slot_time,
            company=company,
            notes=notes,
        )
        if res.get("ok"):
            self._emit_webhook("appointment.booked", res["appointment"])
        return res
