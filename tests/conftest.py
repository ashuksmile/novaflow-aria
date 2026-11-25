"""Shared fixtures. Models load once per test session (CPU, ~3 s)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
OUTPUT = ROOT / "tests" / "output"  # WAVs written here so a human can listen to them


@pytest.fixture(scope="session")
def tts():
    from app.tts import LocalTTSEngine

    engine = LocalTTSEngine()
    assert set(engine.initialize()) == {"en", "hi"}, "Piper voices missing in models/piper/"
    return engine


@pytest.fixture(scope="session")
def stt():
    from app.stt import LocalSTTEngine

    engine = LocalSTTEngine()
    engine.initialize()
    return engine


@pytest.fixture(scope="session")
def human_clips():
    manifest = FIXTURES / "manifest.json"
    if not manifest.exists():
        pytest.skip("Run `python -m scripts.fetch_test_audio` to download real speech fixtures")
    return json.loads(manifest.read_text())


@pytest.fixture(scope="session")
def output_dir():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    return OUTPUT
