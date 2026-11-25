const MIC_WORKLET_CODE = `
  class Mic16k extends AudioWorkletProcessor {
    constructor() {
      super();
      this.ratio = sampleRate / 16000;
      this.pos = 0;
      this.acc = 0;
      this.cnt = 0;
      this.buf = new Int16Array(512);
      this.n = 0;
      this.sq = 0;
    }
    process(inputs) {
      const ch = inputs[0] && inputs[0][0];
      if (!ch) return true;
      for (let i = 0; i < ch.length; i++) {
        this.acc += ch[i];
        this.cnt++;
        this.pos += 1;
        if (this.pos >= this.ratio) {
          this.pos -= this.ratio;
          const v = Math.max(-1, Math.min(1, this.acc / this.cnt));
          this.acc = 0;
          this.cnt = 0;
          this.buf[this.n++] = v < 0 ? v * 32768 : v * 32767;
          this.sq += v * v;
          if (this.n === 512) {
            this.port.postMessage({
              pcm: this.buf.buffer.slice(0),
              rms: Math.sqrt(this.sq / 512),
            });
            this.n = 0;
            this.sq = 0;
          }
        }
      }
      return true;
    }
  }
  registerProcessor('mic-16k', Mic16k);
`;

export interface MicFrame {
  pcm: ArrayBuffer;
  rms: number;
}

export class VoiceAudioEngine {
  private audioCtx: AudioContext | null = null;
  private workletReady = false;
  private micStream: MediaStream | null = null;
  private micSource: MediaStreamAudioSourceNode | null = null;
  private micNode: AudioWorkletNode | null = null;
  private sinkGain: GainNode | null = null;
  private outAnalyser: AnalyserNode | null = null;
  private playHead = 0;
  private playing: AudioBufferSourceNode[] = [];
  private dropTurn = -1;
  public micLevel = 0;

  async ensureContext(): Promise<AudioContext> {
    if (!this.audioCtx) {
      const Ctx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.audioCtx = new Ctx();
      this.outAnalyser = this.audioCtx.createAnalyser();
      this.outAnalyser.fftSize = 512;
      this.outAnalyser.connect(this.audioCtx.destination);
    }
    if (this.audioCtx.state === 'suspended') {
      await this.audioCtx.resume();
    }
    return this.audioCtx;
  }

  async startMic(onFrame: (frame: MicFrame) => void): Promise<void> {
    const ctx = await this.ensureContext();
    this.micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    if (!this.workletReady) {
      const url = URL.createObjectURL(
        new Blob([MIC_WORKLET_CODE], { type: 'application/javascript' })
      );
      await ctx.audioWorklet.addModule(url);
      this.workletReady = true;
    }
    this.micSource = ctx.createMediaStreamSource(this.micStream);
    this.micNode = new AudioWorkletNode(ctx, 'mic-16k');
    this.sinkGain = ctx.createGain();
    this.sinkGain.gain.value = 0; // keep worklet pulled, silent locally
    this.micSource.connect(this.micNode);
    this.micNode.connect(this.sinkGain);
    this.sinkGain.connect(ctx.destination);
    this.micNode.port.onmessage = (ev: MessageEvent<MicFrame>) => {
      this.micLevel = ev.data.rms;
      onFrame(ev.data);
    };
    this.dropTurn = -1;
  }

  stopMic(): void {
    if (this.micStream) this.micStream.getTracks().forEach((t) => t.stop());
    if (this.micNode) {
      this.micNode.port.onmessage = null;
      this.micNode.disconnect();
    }
    if (this.micSource) this.micSource.disconnect();
    this.micStream = null;
    this.micNode = null;
    this.micSource = null;
    this.micLevel = 0;
  }

  setMuted(muted: boolean): void {
    if (this.micStream) {
      this.micStream.getAudioTracks().forEach((t) => {
        t.enabled = !muted;
      });
    }
  }

  markTurnInterrupted(turn: number): void {
    this.dropTurn = Math.max(this.dropTurn, turn);
    this.stopPlayback();
  }

  playAgentAudio(buf: ArrayBuffer): void {
    const view = new DataView(buf);
    const turn = view.getUint32(0, true);
    const rate = view.getUint32(4, true);
    if (turn <= this.dropTurn || !this.audioCtx || !this.outAnalyser) return;

    const pcm = new Int16Array(buf, 8, (buf.byteLength - 8) >> 1);
    const audioBuf = this.audioCtx.createBuffer(1, pcm.length, rate);
    const ch = audioBuf.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;

    const src = this.audioCtx.createBufferSource();
    src.buffer = audioBuf;
    src.connect(this.outAnalyser);
    const startAt = Math.max(this.audioCtx.currentTime + 0.04, this.playHead);
    src.start(startAt);
    this.playHead = startAt + audioBuf.duration;
    this.playing.push(src);
    src.onended = () => {
      this.playing = this.playing.filter((s) => s !== src);
    };
  }

  stopPlayback(): void {
    for (const s of this.playing) {
      try {
        s.stop();
      } catch {
        // already stopped
      }
    }
    this.playing = [];
    this.playHead = 0;
  }

  getOutputLevel(): number {
    if (!this.outAnalyser) return 0;
    const data = new Uint8Array(this.outAnalyser.fftSize);
    this.outAnalyser.getByteTimeDomainData(data);
    let sq = 0;
    for (let i = 0; i < data.length; i++) {
      const x = (data[i] - 128) / 128;
      sq += x * x;
    }
    return Math.sqrt(sq / data.length) * 4.5;
  }
}
