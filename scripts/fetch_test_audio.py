"""Download a handful of REAL human speech clips (with reference transcripts) for STT tests.

English: LibriSpeech dev-clean (via hf-internal-testing/librispeech_asr_dummy)
Hindi:   Google FLEURS hi_in validation (native speakers reading Hindi sentences)

Clips are decoded (PyAV) to 16 kHz mono WAV under tests/fixtures/ with manifest.json.
Run once:  .venv/bin/python scripts/fetch_test_audio.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import urllib.parse
import urllib.request

import av
import numpy as np

from app.audio_utils import float32_to_wav_bytes, resample

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
API = "https://datasets-server.huggingface.co/rows"
SOURCES = [
    # (dataset, config, split, row indices, language, text field)
    ("hf-internal-testing/librispeech_asr_dummy", "clean", "validation", [1, 6, 8, 10], "en", "text"),
    ("google/fleurs", "hi_in", "validation", [3, 1, 6], "hi", "raw_transcription"),
]


def decode(blob: bytes) -> np.ndarray:
    with av.open(io.BytesIO(blob)) as container:
        stream = container.streams.audio[0]
        chunks, rate = [], stream.rate
        for frame in container.decode(stream):
            arr = frame.to_ndarray().astype(np.float32)
            if arr.dtype.kind == "i" or np.abs(arr).max() > 1.5:
                arr = arr / 32768.0
            chunks.append(arr.mean(axis=0) if arr.ndim > 1 else arr)
    return resample(np.concatenate(chunks), rate, 16000)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for dataset, config, split, rows, lang, field in SOURCES:
        for idx in rows:
            q = urllib.parse.urlencode(dict(dataset=dataset, config=config, split=split, offset=idx, length=1))
            row = json.load(urllib.request.urlopen(f"{API}?{q}", timeout=60))["rows"][0]["row"]
            audio_url = row["audio"][0]["src"]
            audio = decode(urllib.request.urlopen(audio_url, timeout=120).read())
            name = f"{lang}_{dataset.split('/')[-1]}_{idx}.wav"
            (OUT / name).write_bytes(float32_to_wav_bytes(audio, 16000))
            text = row[field].strip()
            manifest.append({"file": name, "language": lang, "text": text,
                             "source": f"{dataset}/{config}/{split}#{idx}", "seconds": round(len(audio) / 16000, 2)})
            print(f"{name}: {len(audio) / 16000:.1f}s  {text[:70]}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
