import { useEffect, useRef } from 'react';
import type { CallSummary, Lead, RagSource } from '../types';

const SENTIMENT_CLASS: Record<string, string> = {
  Positive: 'pos',
  Frustrated: 'neg',
  Negative: 'neg',
  Hesitant: 'hes',
};

interface LeadSidePanelProps {
  lead: Lead;
  ragSources: RagSource[];
  summary: CallSummary | null;
  summarizing: boolean;
  onSummarize: () => void;
}

export function LeadSidePanel({
  lead,
  ragSources,
  summary,
  summarizing,
  onSummarize,
}: LeadSidePanelProps) {
  const summaryRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (summary) {
      summaryRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [summary]);

  const score = Math.max(0, Math.min(100, lead.lead_score || 0));
  const sentLabel = lead.sentiment_label || 'Neutral';
  const sentNum = Number(lead.sentiment_score || 0);
  const identity = [lead.name, lead.company, lead.email].filter(Boolean).join(' · ');

  const rows: Array<{ id: string; label: string; value?: string; empty?: string }> = [
    { id: 'lead-identity', label: 'Prospect', value: identity, empty: 'Unknown' },
    { id: 'bant-authority', label: 'Authority', value: lead.authority || lead.role },
    { id: 'bant-budget', label: 'Budget', value: lead.budget },
    { id: 'bant-timeline', label: 'Timeline', value: lead.timeline },
    { id: 'bant-need', label: 'Need', value: lead.need },
  ];

  return (
    <aside className="col-side">
      <div className="panel">
        <div className="panel-head mono">
          <span>
            <span className="yellow">LEAD</span> / QUALIFICATION
          </span>
          <span id="lead-stage" className="tag">
            {lead.qualification_stage || 'New'}
          </span>
        </div>

        <div className="score-row">
          <div className="score">
            <span id="lead-score">{score}</span>
            <small>/100</small>
          </div>
        </div>
        <div className="bar">
          <i id="lead-score-bar" style={{ width: `${score}%` }} />
        </div>

        <div className="kv" style={{ marginTop: 22 }}>
          <span className="mono">Sentiment</span>
          <span className="v">
            <span
              id="sentiment-label"
              className={`label ${SENTIMENT_CLASS[sentLabel] || ''}`}
            >
              {sentLabel}
            </span>{' '}
            <span id="sentiment-score" className="mono faint">
              {(sentNum >= 0 ? '+' : '') + sentNum.toFixed(2)}
            </span>
          </span>
        </div>

        {rows.map((r) => (
          <div key={r.id} className="kv">
            <span className="mono">{r.label}</span>
            <span id={r.id} className={`v${r.value ? '' : ' muted'}`}>
              {r.value || r.empty || '—'}
            </span>
          </div>
        ))}

        <div className="tags" id="objections-list">
          {(lead.objections || []).map((o) => (
            <span key={o} className="tag red">
              Objection / {o}
            </span>
          ))}
        </div>
      </div>

      {ragSources.length > 0 && (
        <div className="panel" id="rag-section">
          <div className="panel-head mono">
            <span>
              <span className="yellow">SOURCES</span> / KNOWLEDGE BASE
            </span>
          </div>
          <div id="rag-panel">
            {ragSources.map((s, idx) => (
              <div key={s.chunk_id || `${s.heading}-${idx}`} className="source">
                <div className="src-head">
                  <span className="tag">Source / {String(idx + 1).padStart(2, '0')}</span>
                  <span className="mono faint">SIM {s.score}</span>
                </div>
                <h4>{s.heading}</h4>
                <p>{s.content}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="panel summary">
        <div className="panel-head mono">
          <span>
            <span className="yellow">SUMMARY</span> / POST-CALL
          </span>
          <span id="summary-trend">{summary?.sentiment_trend || ''}</span>
        </div>

        {summary && (
          <div id="summary-card" ref={summaryRef}>
            <h4 id="summary-text">{summary.summary}</h4>
            <div className="mono faint">Next action</div>
            <p id="summary-action">{summary.recommended_action}</p>
            <div className="mono faint">Follow-up email draft</div>
            <pre id="summary-email">{summary.followup_email}</pre>
          </div>
        )}

        <button
          id="btn-summarize"
          className="btn-line mono"
          style={{ marginTop: 12 }}
          disabled={summarizing}
          onClick={onSummarize}
        >
          <span>{summarizing ? 'Summarizing…' : 'Summarize call'}</span>
          <span>↗</span>
        </button>
      </div>
    </aside>
  );
}
