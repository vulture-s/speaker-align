# -*- coding: utf-8 -*-
"""The factory + the 'alignment works without pyannote installed' contract."""
import pytest

import speaker_align
from speaker_align import get_diarizer


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown diarization backend"):
        get_diarizer("not-a-backend")


def test_explicit_token_beats_env(monkeypatch):
    monkeypatch.setenv("SPEAKER_ALIGN_AUTH_TOKEN", "from-env")
    diarizer = get_diarizer("pyannote", auth_token="explicit")
    assert diarizer._auth_token == "explicit"


def test_env_fallback_order(monkeypatch):
    for name in speaker_align.AUTH_TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYANNOTE_AUTH_TOKEN", "second-choice")
    assert get_diarizer("pyannote")._auth_token == "second-choice"


def test_missing_token_yields_empty_not_none(monkeypatch):
    for name in speaker_align.AUTH_TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    assert get_diarizer("pyannote")._auth_token == ""


def test_constructing_a_diarizer_does_not_load_the_pipeline(monkeypatch):
    """Building one must stay cheap and side-effect free — no network, no weights."""
    for name in speaker_align.AUTH_TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    diarizer = get_diarizer("pyannote")  # would raise here if it loaded eagerly
    assert diarizer._pipeline is None


def test_alignment_half_needs_no_backend_import():
    """The whole point of the split: importing align must not import pyannote."""
    import importlib
    import sys

    for module in [m for m in sys.modules if m.startswith("speaker_align")]:
        del sys.modules[module]
    importlib.import_module("speaker_align.align")
    assert not any(m.startswith("pyannote") for m in sys.modules)
