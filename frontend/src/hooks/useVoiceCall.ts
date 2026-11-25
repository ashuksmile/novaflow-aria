import { useCallback, useEffect, useRef, useState } from 'react';
import { summarizeCallSession } from '../lib/api';
import { VoiceAudioEngine } from '../lib/audio';
import type {
  CallState,
  CallSummary,
  LanguageMode,
  LatencyMetrics,
  Lead,
  RagSource,
  ServerWsEvent,
  TranscriptEntry,
} from '../types';

const DEFAULT_LEAD: Lead = {
  lead_score: 10,
  qualification_stage: 'New',
  sentiment_label: 'Neutral',
  sentiment_score: 0,
  objections: [],
};

function makeSessionId(): string {
  return 'call-' + Math.random().toString(36).slice(2, 8);
}

export interface MicHelpInfo {
  title: string;
  body: string;
  httpsUrl?: string;
}

export function useVoiceCall(onDataUpdated?: () => void) {
  const [sessionId, setSessionId] = useState<string>(makeSessionId);
  const [wsStatus, setWsStatus] = useState<{ label: string; color: string }>({
    label: 'Connecting',
    color: 'var(--green)',
  });
  const [inCall, setInCall] = useState(false);
  const [muted, setMuted] = useState(false);
  const [callState, setCallState] = useState<CallState>('idle');
  const [language, setLanguageState] = useState<LanguageMode>('auto');
  const [bargeIn, setBargeInState] = useState(true);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [caption, setCaption] = useState('');
  const [micHelp, setMicHelp] = useState<MicHelpInfo | null>(null);
  const [meterBars, setMeterBars] = useState<number[]>(() => new Array(18).fill(0));
  const [metrics, setMetrics] = useState<LatencyMetrics>({});
  const [entries, setEntries] = useState<TranscriptEntry[]>([]);
  const [turnCount, setTurnCount] = useState(0);
  const [lead, setLead] = useState<Lead>(DEFAULT_LEAD);
  const [ragSources, setRagSources] = useState<RagSource[]>([]);
  const [summary, setSummary] = useState<CallSummary | null>(null);
  const [summarizing, setSummarizing] = useState(false);

  const audioRef = useRef<VoiceAudioEngine | null>(null);
  if (!audioRef.current) audioRef.current = new VoiceAudioEngine();

  const wsRef = useRef<WebSocket | null>(null);
  const inCallRef = useRef(inCall);
  const mutedRef = useRef(muted);
  const callStateRef = useRef(callState);
  const languageRef = useRef(language);
  const bargeInRef = useRef(bargeIn);
  const turnCountRef = useRef(turnCount);
  const sessionIdRef = useRef(sessionId);
  const onDataUpdatedRef = useRef(onDataUpdated);
  const meterHistRef = useRef<number[]>(new Array(18).fill(0));

  inCallRef.current = inCall;
  mutedRef.current = muted;
  callStateRef.current = callState;
  languageRef.current = language;
  bargeInRef.current = bargeIn;
  turnCountRef.current = turnCount;
  sessionIdRef.current = sessionId;
  onDataUpdatedRef.current = onDataUpdated;

  const wsSend = useCallback((payload: Record<string, unknown>) => {
    const sock = wsRef.current;
    if (sock && sock.readyState === WebSocket.OPEN) {
      sock.send(JSON.stringify(payload));
    }
  }, []);

  const addSystemNote = useCallback((message: string) => {
    setEntries((prev) => [
      ...prev,
      { id: `sys-${Date.now()}-${Math.random()}`, kind: 'system', text: message },
    ]);
  }, []);

  const generateSummary = useCallback(async () => {
    setSummarizing(true);
    try {
      const data = await summarizeCallSession(sessionIdRef.current);
      setSummary(data.summary);
      if (data.lead) setLead(data.lead);
      onDataUpdatedRef.current?.();
    } catch (err) {
      addSystemNote(`Summary failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSummarizing(false);
    }
  }, [addSystemNote]);

  const stopCallInternal = useCallback(
    (shouldSummarize: boolean) => {
      setInCall(false);
      setMuted(false);
      wsSend({ type: 'end_call' });
      audioRef.current?.stopPlayback();
      audioRef.current?.stopMic();
      meterHistRef.current.fill(0);
      setMeterBars(new Array(18).fill(0));
      setCallState('idle');
      setCaption('');
      if (shouldSummarize && turnCountRef.current > 0) {
        void generateSummary();
      }
    },
    [generateSummary, wsSend]
  );

  // Expose playAgentAudio on window for automated Playwright voice verification
  useEffect(() => {
    const audio = audioRef.current!;
    window.playAgentAudio = (buf: ArrayBuffer) => audio.playAgentAudio(buf);
    return () => {
      audio.stopPlayback();
      audio.stopMic();
    };
  }, []);

  // Keep body data attributes and window.callState in sync
  useEffect(() => {
    document.body.dataset.state = callState;
    window.callState = callState;
  }, [callState]);

  useEffect(() => {
    document.body.classList.toggle('in-call', inCall);
  }, [inCall]);

  // Call duration timer
  useEffect(() => {
    if (!inCall) return;
    const started = Date.now();
    setElapsedSec(0);
    const id = window.setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - started) / 1000));
    }, 500);
    return () => window.clearInterval(id);
  }, [inCall]);

  // Audio-reactive arc level animation loop
  useEffect(() => {
    let rafId = 0;
    const tick = () => {
      let level = 0;
      const state = callStateRef.current;
      const audio = audioRef.current;
      if (audio && state === 'speaking') {
        level = audio.getOutputLevel();
      } else if (audio && inCallRef.current && !mutedRef.current) {
        level = audio.micLevel * 7;
      } else if (state === 'thinking') {
        level = 0.25 + 0.15 * Math.sin(performance.now() / 260);
      }
      level = Math.max(0, Math.min(1, level));
      document.documentElement.style.setProperty('--level', level.toFixed(3));
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, []);

  const handleServerEvent = useCallback(
    (msg: ServerWsEvent) => {
      switch (msg.type) {
        case 'connected':
          setLead(msg.lead);
          break;
        case 'state':
          setCallState(msg.state);
          if (msg.state === 'listening') setCaption('');
          if (msg.state !== 'speaking') {
            setEntries((prev) => prev.map((e) => (e.live ? { ...e, live: false } : e)));
          }
          break;
        case 'transcript':
          if (msg.stt_ms) {
            setMetrics((m) => ({ ...m, stt_ms: msg.stt_ms }));
          }
          setTurnCount((n) => n + 1);
          setEntries((prev) => [
            ...prev,
            {
              id: `u-${msg.turn}-${Date.now()}`,
              kind: 'user',
              turn: msg.turn,
              text: msg.text,
              language: msg.language || 'en',
              typed: msg.typed,
            },
          ]);
          break;
        case 'agent_sentence':
          setCaption(msg.text);
          setEntries((prev) => {
            const idx = prev.findIndex((e) => e.kind === 'agent' && e.turn === msg.turn);
            if (idx === -1) {
              return [
                ...prev,
                {
                  id: `a-${msg.turn}`,
                  kind: 'agent',
                  turn: msg.turn,
                  text: msg.text,
                  language: msg.language || 'en',
                  live: true,
                  tools: [],
                },
              ];
            }
            const next = [...prev];
            const cur = next[idx];
            next[idx] = {
              ...cur,
              text: `${cur.text} ${msg.text}`.trim(),
              language: msg.language || cur.language,
              live: true,
            };
            return next;
          });
          break;
        case 'tool':
          setEntries((prev) => {
            const idx = prev.findIndex((e) => e.kind === 'agent' && e.turn === msg.turn);
            const tag = { tool: msg.tool, status: msg.status };
            if (idx === -1) {
              return [
                ...prev,
                {
                  id: `a-${msg.turn}`,
                  kind: 'agent',
                  turn: msg.turn,
                  text: '',
                  language: 'en',
                  tools: [tag],
                },
              ];
            }
            const next = [...prev];
            const cur = next[idx];
            const tools = [...(cur.tools || [])];
            const tIdx = tools.findIndex((t) => t.tool === msg.tool);
            if (tIdx === -1) tools.push(tag);
            else tools[tIdx] = tag;
            next[idx] = { ...cur, tools };
            return next;
          });
          break;
        case 'agent_done':
          if (msg.rag_sources && msg.rag_sources.length > 0) {
            setRagSources(msg.rag_sources);
          }
          setEntries((prev) => {
            const idx = prev.findIndex((e) => e.kind === 'agent' && e.turn === msg.turn);
            if (idx === -1) {
              return [
                ...prev,
                {
                  id: `a-${msg.turn}`,
                  kind: 'agent',
                  turn: msg.turn,
                  text: msg.text,
                  language: msg.language || 'en',
                  live: false,
                },
              ];
            }
            const next = [...prev];
            const cur = next[idx];
            next[idx] = {
              ...cur,
              text: cur.interrupted ? cur.text : msg.text,
              language: msg.language || cur.language,
              live: false,
            };
            return next;
          });
          onDataUpdatedRef.current?.();
          break;
        case 'interrupt':
          audioRef.current?.markTurnInterrupted(msg.turn);
          setEntries((prev) =>
            prev.map((e) =>
              e.kind === 'agent' && e.turn === msg.turn
                ? { ...e, live: false, interrupted: true }
                : e
            )
          );
          break;
        case 'metrics':
          setMetrics((m) => ({
            stt_ms: msg.stt_ms ?? m.stt_ms,
            llm_first_token_ms: msg.llm_first_token_ms ?? m.llm_first_token_ms,
            tts_first_ms: msg.tts_first_ms ?? m.tts_first_ms,
            voice_to_voice_ms: msg.voice_to_voice_ms ?? m.voice_to_voice_ms,
          }));
          break;
        case 'intelligence':
          setLead(msg.lead);
          onDataUpdatedRef.current?.();
          break;
        case 'info':
          addSystemNote(msg.message);
          break;
      }
    },
    [addSystemNote]
  );

  const connectWebSocket = useCallback(
    (sid: string) => {
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const sock = new WebSocket(`${proto}//${window.location.host}/ws/voice/${sid}`);
      sock.binaryType = 'arraybuffer';
      wsRef.current = sock;
      window.ws = sock;

      sock.onopen = () => setWsStatus({ label: 'Ready', color: 'var(--green)' });
      sock.onclose = () => {
        setWsStatus({ label: 'Offline', color: 'var(--red)' });
        if (inCallRef.current) stopCallInternal(false);
      };
      sock.onerror = () => setWsStatus({ label: 'Error', color: 'var(--red)' });
      sock.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          if (window.playAgentAudio) window.playAgentAudio(event.data);
          else audioRef.current?.playAgentAudio(event.data);
          return;
        }
        handleServerEvent(JSON.parse(event.data) as ServerWsEvent);
      };
      return sock;
    },
    [handleServerEvent, stopCallInternal]
  );

  // Open WebSocket whenever sessionId changes
  useEffect(() => {
    const sock = connectWebSocket(sessionId);
    return () => {
      sock.onclose = null;
      sock.close();
    };
  }, [sessionId, connectWebSocket]);

  const startCall = useCallback(async () => {
    setMicHelp(null);
    if (!window.isSecureContext || !navigator.mediaDevices) {
      setMicHelp({
        title: 'Your browser blocks the microphone on plain http://.',
        body: 'Only https:// or localhost may use it. Open the HTTPS URL below (accept the self-signed certificate), or type below — Aria still answers by voice.',
        httpsUrl: `https://${window.location.hostname}:8443${window.location.pathname}`,
      });
      return;
    }

    try {
      let sock = wsRef.current;
      if (!sock || sock.readyState !== WebSocket.OPEN) {
        sock = connectWebSocket(sessionIdRef.current);
      }
      await audioRef.current!.startMic((frame) => {
        const hist = meterHistRef.current;
        hist.shift();
        hist.push(mutedRef.current ? 0 : frame.rms);
        setMeterBars([...hist]);
        const s = wsRef.current;
        if (!mutedRef.current && s && s.readyState === WebSocket.OPEN) {
          s.send(frame.pcm);
        }
      });

      setInCall(true);
      setMuted(false);
      setCaption('');

      const sendStart = () => {
        wsSend({
          type: 'start_call',
          language: languageRef.current,
          barge_in: bargeInRef.current,
        });
      };
      if (sock.readyState === WebSocket.OPEN) {
        sendStart();
      } else {
        sock.addEventListener('open', sendStart, { once: true });
      }
    } catch (err) {
      const name = err instanceof Error ? err.name : '';
      const message = err instanceof Error ? err.message : String(err);
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        setMicHelp({
          title: 'Microphone permission was denied.',
          body: 'Click the lock / site-settings icon in the address bar, allow the microphone, then press Start call again.',
        });
      } else {
        setMicHelp({
          title: `Microphone unavailable (${message}).`,
          body: 'Check that a mic is connected, or type below.',
        });
      }
    }
  }, [connectWebSocket, wsSend]);

  const toggleCall = useCallback(() => {
    if (inCallRef.current) stopCallInternal(true);
    else void startCall();
  }, [startCall, stopCallInternal]);

  const toggleMute = useCallback(() => {
    if (!inCallRef.current) return;
    setMuted((prev) => {
      const next = !prev;
      audioRef.current?.setMuted(next);
      return next;
    });
  }, []);

  const setLanguage = useCallback(
    (next: LanguageMode) => {
      setLanguageState(next);
      wsSend({ type: 'config', language: next, barge_in: bargeInRef.current });
    },
    [wsSend]
  );

  const setBargeIn = useCallback(
    (next: boolean) => {
      setBargeInState(next);
      wsSend({ type: 'config', language: languageRef.current, barge_in: next });
    },
    [wsSend]
  );

  const sendText = useCallback(
    async (text: string) => {
      const clean = text.trim();
      if (!clean) return;
      await audioRef.current!.ensureContext();
      const sock = wsRef.current;
      if (!sock || sock.readyState !== WebSocket.OPEN) {
        addSystemNote('Not connected — reconnecting, try again in a second.');
        connectWebSocket(sessionIdRef.current);
        return;
      }
      wsSend({ type: 'text', text: clean, language: languageRef.current });
    },
    [addSystemNote, connectWebSocket, wsSend]
  );

  const newSession = useCallback(() => {
    if (inCallRef.current) stopCallInternal(false);
    setEntries([]);
    setTurnCount(0);
    setSummary(null);
    setRagSources([]);
    setMetrics({});
    setElapsedSec(0);
    setCaption('');
    setMicHelp(null);
    setLead(DEFAULT_LEAD);
    setSessionId(makeSessionId());
  }, [stopCallInternal]);

  return {
    sessionId,
    wsStatus,
    inCall,
    muted,
    callState,
    language,
    bargeIn,
    elapsedSec,
    caption,
    micHelp,
    meterBars,
    metrics,
    entries,
    turnCount,
    lead,
    ragSources,
    summary,
    summarizing,
    toggleCall,
    toggleMute,
    setLanguage,
    setBargeIn,
    sendText,
    newSession,
    generateSummary,
  };
}
