import type { CallState, LanguageMode, TabId } from '../types';

const STATE_LABELS: Record<CallState, { label: string; color: string }> = {
  idle: { label: 'Ready', color: 'var(--green)' },
  listening: { label: 'Listening', color: 'var(--cyan)' },
  user_speaking: { label: "You're speaking", color: 'var(--green)' },
  transcribing: { label: 'Transcribing', color: 'var(--yellow)' },
  thinking: { label: 'Thinking', color: 'var(--violet)' },
  speaking: { label: 'Aria is speaking', color: 'var(--orange)' },
};

interface NavbarProps {
  activeTab: TabId;
  onSelectTab: (tab: TabId) => void;
  turnCount: number;
  demoCount: number;
  leadCount: number;
  summaryCount: number;
  language: LanguageMode;
  onLanguageChange: (lang: LanguageMode) => void;
  onNewSession: () => void;
  callState: CallState;
  inCall: boolean;
  wsStatus: { label: string; color: string };
}

export function Navbar({
  activeTab,
  onSelectTab,
  turnCount,
  demoCount,
  leadCount,
  summaryCount,
  language,
  onLanguageChange,
  onNewSession,
  callState,
  inCall,
  wsStatus,
}: NavbarProps) {
  const status =
    callState === 'idle'
      ? inCall
        ? { label: 'Connecting', color: 'var(--green)' }
        : wsStatus
      : STATE_LABELS[callState];

  const tabs: Array<{ id: TabId; label: string; count: number; countId: string }> = [
    { id: 'voice', label: 'Call', count: turnCount, countId: 'turn-count' },
    { id: 'calendar', label: 'Calendar', count: demoCount, countId: 'demo-count' },
    { id: 'crm', label: 'CRM', count: leadCount, countId: 'lead-count' },
    { id: 'insights', label: 'Insights', count: summaryCount, countId: 'summary-count' },
  ];

  return (
    <header className="nav">
      <a className="brand" href="/" aria-label="NovaFlow home">
        <svg viewBox="0 0 38 20" fill="none" aria-hidden="true">
          <path d="M2 19a17 17 0 0 1 34 0" stroke="currentColor" strokeWidth="4.5" />
          <path d="M11 19a8 8 0 0 1 16 0" stroke="currentColor" strokeWidth="4.5" />
        </svg>
        <span>NOVAFLOW</span>
      </a>

      <nav className="tabs" role="tablist">
        {tabs.map((t) => (
          <button
            key={t.id}
            id={`tab-${t.id}`}
            className={`tab${activeTab === t.id ? ' active' : ''}`}
            role="tab"
            aria-selected={activeTab === t.id}
            onClick={() => onSelectTab(t.id)}
          >
            {t.label}{' '}
            <span className="br">
              [<span id={t.countId}>{t.count}</span>]
            </span>
          </button>
        ))}
      </nav>

      <div className="nav-right">
        <label className="sr-only" htmlFor="lang-select">
          Language
        </label>
        <select
          id="lang-select"
          className="select mono"
          value={language}
          onChange={(e) => onLanguageChange(e.target.value as LanguageMode)}
        >
          <option value="auto">EN / HI — Auto</option>
          <option value="en">English</option>
          <option value="hi">हिन्दी — Hindi</option>
        </select>

        <button className="link-btn" onClick={onNewSession}>
          New session
        </button>

        <span id="status-badge" className="status mono">
          <span className="dot" style={{ background: status.color, color: status.color }} />
          <span className="txt">{status.label}</span>
        </span>
      </div>
    </header>
  );
}
