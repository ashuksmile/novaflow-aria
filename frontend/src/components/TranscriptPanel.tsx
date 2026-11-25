import { type FormEvent, useEffect, useRef, useState } from 'react';
import type { TranscriptEntry } from '../types';

const SAMPLE_PROMPTS: ReadonlyArray<readonly [string, string]> = [
  ['EN / PRICING', 'What does the Growth plan cost for twenty sales reps?'],
  ['HI / SECURITY', 'क्या आपका डेटा भारत में स्टोर होता है?'],
  ['EN / BOOKING', 'Book a demo tomorrow at 11:30 AM. My email is rahul@acmetech.io'],
];

interface TranscriptPanelProps {
  entries: TranscriptEntry[];
  turnCount: number;
  inCall: boolean;
  onSendText: (text: string) => void;
}

export function TranscriptPanel({
  entries,
  turnCount,
  inCall,
  onSendText,
}: TranscriptPanelProps) {
  const [draft, setDraft] = useState('');
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = logRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 140;
    if (nearBottom) el.scrollTop = el.scrollHeight;
  }, [entries]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const clean = draft.trim();
    if (!clean) return;
    setDraft('');
    onSendText(clean);
  };

  return (
    <div className="record">
      <div className="section-head mono">
        <span>
          <span className="yellow">THE RECORD</span> / LIVE TRANSCRIPT
        </span>
        <span id="record-count">
          {turnCount} {turnCount === 1 ? 'TURN' : 'TURNS'}
        </span>
      </div>

      <div id="chat-log" className="log" ref={logRef}>
        {entries.length === 0 && !inCall ? (
          <div className="empty-state" id="empty-state">
            <h3>Press start call and talk like you would on the phone.</h3>
            <div className="mono faint" style={{ marginBottom: 8 }}>
              Try saying
            </div>
            <div className="try-list">
              {SAMPLE_PROMPTS.map(([tag, text]) => (
                <button key={tag} className="try" onClick={() => onSendText(text)}>
                  <span className="mono">{tag}</span>
                  <span className="txt">{text}</span>
                  <span className="arr">↗</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          entries.map((entry) => {
            if (entry.kind === 'system') {
              return (
                <div key={entry.id} className="sys mono">
                  {entry.text}
                </div>
              );
            }
            const classes = [
              'entry',
              entry.kind,
              entry.live ? 'live' : '',
              entry.interrupted ? 'interrupted' : '',
            ]
              .filter(Boolean)
              .join(' ');
            return (
              <div key={entry.id} className={classes}>
                <div className="entry-meta mono">
                  <span className="who">{entry.kind === 'user' ? 'Prospect' : 'Aria'}</span>
                  <span>
                    {entry.language || 'en'}
                    {entry.typed ? ' · typed' : ''}
                  </span>
                </div>
                <div>
                  <div className="body">{entry.text}</div>
                  <div className="tags">
                    {(entry.tools || []).map((t) => (
                      <span
                        key={t.tool}
                        className={`tag${t.status === 'running' ? ' running' : ''}`}
                      >
                        TOOL / {t.tool}
                        {t.status === 'done' ? '' : ' …'}
                      </span>
                    ))}
                    {entry.interrupted && (
                      <span className="tag red">Interrupted by caller</span>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      <form className="composer" onSubmit={handleSubmit}>
        <span className="mono">Type /</span>
        <input
          id="text-input"
          type="text"
          autoComplete="off"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Noisy room? Type here — Aria still answers by voice"
        />
        <button type="submit" className="mono">
          Send ↗
        </button>
      </form>
    </div>
  );
}
