import type { CalendarData, CallSummary, DashboardData, Lead } from '../types';

export async function fetchDashboard(): Promise<DashboardData> {
  const res = await fetch('/api/dashboard');
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchCalendar(): Promise<CalendarData> {
  const res = await fetch('/api/calendar');
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function summarizeCallSession(
  sessionId: string
): Promise<{ summary: CallSummary; lead?: Lead }> {
  const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/summarize`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
