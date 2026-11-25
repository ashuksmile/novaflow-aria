import type { MicHelpInfo } from '../hooks/useVoiceCall';
import type { CallState, LatencyMetrics } from '../types';

const STATE_COPY: Record<CallState, { title: string; hint: string }> = {
  idle: { title: 'Ready when you are', hint: 'Start a call and just talk' },
  listening: {
    title: 'Listening',
    hint: 'Just start talking — 650 ms of silence ends your turn',
  },
  user_speaking: {
    title: "You're speaking",
    hint: 'VAD hears you — take your time',
  },
  transcribing: { title: 'Transcribing', hint: 'Whisper · local CPU' },
  thinking: { title: 'Thinking', hint: 'Groq · gpt-oss-120b · RAG + tools' },
  speaking: { title: 'Aria is speaking', hint: '' },
};

function formatTimer(totalSec: number): string {
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

interface CallStageProps {
  sessionId: string;
  elapsedSec: number;
  callState: CallState;
  caption: string;
  inCall: boolean;
  muted: boolean;
  bargeIn: boolean;
  micHelp: MicHelpInfo | null;
  meterBars: number[];
  metrics: LatencyMetrics;
  onToggleCall: () => void;
  onToggleMute: () => void;
  onBargeInChange: (v: boolean) => void;
}

export function CallStage({
  sessionId,
  elapsedSec,
  callState,
  caption,
  inCall,
  muted,
  bargeIn,
  micHelp,
  meterBars,
  metrics,
  onToggleCall,
  onToggleMute,
  onBargeInChange,
}: CallStageProps) {
  const copy = STATE_COPY[callState];
  const hint =
    callState === 'speaking'
      ? bargeIn
        ? 'Talk over her to interrupt'
        : 'Barge-in off — wait for her to finish'
      : copy.hint;

  const hasMetrics =
    metrics.stt_ms !== undefined ||
    metrics.llm_first_token_ms !== undefined ||
    metrics.tts_first_ms !== undefined ||
    metrics.voice_to_voice_ms !== undefined;

  return (
    <div className="stage" id="stage">
      <div className="stage-meta mono">
        <span>
          <span className="yellow">CALL</span> &nbsp;
          <span id="call-timer">{formatTimer(elapsedSec)}</span>
        </span>
        <span className="faint" id="session-label">
          {sessionId.toUpperCase()}
        </span>
      </div>

      <div className="stage-body">
        <div>
          <h1 className="state-title" aria-live="polite">
            <span id="call-state">{copy.title}</span>
            <span className="cursor" />
          </h1>
          <div id="state-hint" className="state-hint mono">
            {hint}
          </div>
          <p id="live-caption" className={`caption${caption ? '' : ' empty'}`}>
            {caption}
          </p>
        </div>

        <div className="arc-wrap" aria-hidden="true">
          <svg viewBox="0 0 400 320">
            <defs>
              <linearGradient id="arcGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0" stopColor="#ff7a2f" />
                <stop offset="0.35" stopColor="#f7a24b" />
                <stop offset="0.62" stopColor="#e84bd3" />
                <stop offset="1" stopColor="#3d5bff" />
              </linearGradient>
            </defs>
            <circle
              cx="200"
              cy="230"
              r="170"
              fill="none"
              stroke="rgba(247,248,246,0.10)"
              strokeDasharray="2 7"
            />
            <g className="arc-glow">
              <path
                d="M70 230a130 130 0 0 1 260 0"
                stroke="url(#arcGrad)"
                strokeWidth="46"
                fill="none"
              />
            </g>
            <g className="arc-main">
              <path
                d="M70 230a130 130 0 0 1 260 0"
                stroke="url(#arcGrad)"
                strokeWidth="46"
                fill="none"
              />
              <path
                d="M138 230a62 62 0 0 1 124 0"
                stroke="url(#arcGrad)"
                strokeWidth="30"
                fill="none"
                opacity="0.85"
              />
            </g>
          </svg>
        </div>
      </div>

      {micHelp && (
        <div id="mic-help" className="notice">
          <strong>{micHelp.title}</strong> {micHelp.body}{' '}
          {micHelp.httpsUrl && <a href={micHelp.httpsUrl}>{micHelp.httpsUrl}</a>}
        </div>
      )}

      <div className="controls">
        <button
          id="call-btn"
          className={`btn-primary${inCall ? ' live' : ''}`}
          onClick={onToggleCall}
          title="Shortcut: Space"
        >
          <span id="call-btn-label">{inCall ? 'End call' : 'Start call'}</span>
          <span className="arrow">↗</span>
        </button>

        <button
          id="mute-btn"
          className={`btn-ghost${muted ? ' on' : ''}`}
          onClick={onToggleMute}
          title="Shortcut: M"
          disabled={!inCall}
        >
          {muted ? 'Unmute' : 'Mute'}
        </button>

        <label
          className="toggle mono"
          title="Let the caller interrupt Aria. Use headphones; untick on laptop speakers."
        >
          <input
            id="bargein-toggle"
            type="checkbox"
            checked={bargeIn}
            onChange={(e) => onBargeInChange(e.target.checked)}
          />{' '}
          Barge-in
        </label>

        <span className="meter" id="meter" aria-label="Microphone level">
          {meterBars.map((rms, idx) => (
            <i key={idx} style={{ height: `${2 + Math.min(20, rms * 260)}px` }} />
          ))}
        </span>
      </div>

      <div className={`metrics-line mono${hasMetrics ? '' : ' hidden'}`} id="metrics-line">
        <span>
          STT <b id="metric-stt">{metrics.stt_ms !== undefined ? `${metrics.stt_ms}ms` : '—'}</b>
        </span>
        <span>
          LLM{' '}
          <b id="metric-llm">
            {metrics.llm_first_token_ms !== undefined ? `${metrics.llm_first_token_ms}ms` : '—'}
          </b>
        </span>
        <span>
          TTS{' '}
          <b id="metric-tts">
            {metrics.tts_first_ms !== undefined ? `${metrics.tts_first_ms}ms` : '—'}
          </b>
        </span>
        <span className="yellow">
          Voice→voice{' '}
          <b id="metric-v2v">
            {metrics.voice_to_voice_ms !== undefined ? `${metrics.voice_to_voice_ms}ms` : '—'}
          </b>
        </span>
      </div>
    </div>
  );
}
