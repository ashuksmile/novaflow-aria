export type TabId = 'voice' | 'calendar' | 'crm' | 'insights';

export type LanguageMode = 'auto' | 'en' | 'hi';

export type CallState =
  | 'idle'
  | 'listening'
  | 'user_speaking'
  | 'transcribing'
  | 'thinking'
  | 'speaking';

export interface Lead {
  id?: number;
  session_id?: string;
  name?: string;
  email?: string;
  company?: string;
  role?: string;
  phone?: string;
  language?: string;
  budget?: string;
  authority?: string;
  need?: string;
  timeline?: string;
  lead_score: number;
  qualification_stage: string;
  sentiment_label: string;
  sentiment_score: number;
  objections?: string[];
  notes?: string;
  created_at?: string;
  updated_at?: string;
}

export interface RagSource {
  chunk_id?: string;
  source?: string;
  heading: string;
  content: string;
  score: number;
}

export interface ToolTag {
  tool: string;
  status: 'running' | 'done';
}

export interface TranscriptEntry {
  id: string;
  kind: 'user' | 'agent' | 'system';
  turn?: number;
  text: string;
  language?: string;
  typed?: boolean;
  live?: boolean;
  interrupted?: boolean;
  tools?: ToolTag[];
}

export interface LatencyMetrics {
  stt_ms?: number;
  llm_first_token_ms?: number;
  tts_first_ms?: number;
  voice_to_voice_ms?: number;
}

export interface Appointment {
  id: number;
  session_id: string;
  lead_id?: number;
  lead_name: string;
  lead_email: string;
  company: string;
  slot_date: string;
  slot_time: string;
  status: string;
  notes: string;
  created_at: string;
}

export interface CallSummary {
  id?: number;
  session_id: string;
  summary: string;
  pain_points?: string[];
  objections?: string[];
  bant?: Record<string, string>;
  sentiment_trend: string;
  recommended_action: string;
  followup_email: string;
  lead_score: number;
  created_at?: string;
}

export interface DashboardMetrics {
  total_calls: number;
  qualified_leads: number;
  booked_demos: number;
  avg_lead_score: number;
  avg_sentiment: number;
}

export interface DashboardData {
  metrics: DashboardMetrics;
  leads: Lead[];
  appointments: Appointment[];
  summaries: CallSummary[];
}

export interface CalendarSlot {
  time: string;
  status: 'Open' | 'Booked';
  appointment: Appointment | null;
}

export interface CalendarDay {
  date: string;
  slots: CalendarSlot[];
}

export interface CalendarData {
  days: CalendarDay[];
  appointments: Appointment[];
}

export type ServerWsEvent =
  | { type: 'connected'; session_id: string; lead: Lead; input_sample_rate?: number }
  | { type: 'state'; state: CallState; turn?: number }
  | {
      type: 'transcript';
      turn: number;
      text: string;
      language?: string;
      confidence?: number;
      stt_ms?: number;
      typed?: boolean;
    }
  | { type: 'agent_sentence'; turn: number; text: string; language?: string }
  | { type: 'tool'; turn: number; tool: string; status: 'running' | 'done'; args?: Record<string, unknown> }
  | {
      type: 'agent_done';
      turn: number;
      text: string;
      language?: string;
      rag_sources?: RagSource[];
      metrics?: LatencyMetrics;
    }
  | { type: 'interrupt'; turn: number }
  | ({ type: 'metrics' } & LatencyMetrics)
  | { type: 'intelligence'; lead: Lead }
  | { type: 'info'; message: string; reason?: string };

declare global {
  interface Window {
    ws?: WebSocket | null;
    callState?: CallState;
    playAgentAudio?: (buf: ArrayBuffer) => void;
  }
}
