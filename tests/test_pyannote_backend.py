# -*- coding: utf-8 -*-
"""PyannoteDiarizer against a fake pyannote, so CI needs no torch or weights.

Pins the three seams the 0.2.0 switch to community-1 touched: the token keyword
(4.x ``token`` vs 3.x ``use_auth_token``), the output shape (4.x wraps the
annotation in a DiarizeOutput), and speaker-count hints reaching the pipeline.
"""
import sys
import types

import pytest

from speaker_align import pyannote as backend


class _Turn:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _Annotation:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        for start, end, spk in self._tracks:
            yield _Turn(start, end), None, spk


TRACKS = [(0.0, 1.234, "SPEAKER_00"), (1.5, 3.0, "SPEAKER_01"), (3.0, 4.0, "SPEAKER_00")]


def _install_fake(monkeypatch, *, wrap_output, accepts_token):
    calls = {}

    class _Pipeline:
        def __call__(self, audio, **hints):
            calls["audio"] = audio
            calls["hints"] = hints
            ann = _Annotation(TRACKS)
            return types.SimpleNamespace(speaker_diarization=ann) if wrap_output else ann

        @classmethod
        def from_pretrained(cls, model, **kw):
            if "token" in kw and not accepts_token:
                raise TypeError("unexpected keyword argument 'token'")
            calls["model"] = model
            calls["auth_kw"] = kw
            return cls()

    audio_mod = types.ModuleType("pyannote.audio")
    audio_mod.Pipeline = _Pipeline
    pkg = types.ModuleType("pyannote")
    pkg.audio = audio_mod
    monkeypatch.setitem(sys.modules, "pyannote", pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio_mod)
    return calls


def test_default_model_is_community_1():
    assert backend.MODEL == "pyannote/speaker-diarization-community-1"
    assert backend.PyannoteDiarizer(auth_token="t")._model == backend.MODEL


@pytest.mark.parametrize("wrap_output", [True, False], ids=["4.x-DiarizeOutput", "3.x-Annotation"])
def test_both_output_shapes_give_the_same_segments(monkeypatch, wrap_output):
    _install_fake(monkeypatch, wrap_output=wrap_output, accepts_token=True)
    result = backend.PyannoteDiarizer(auth_token="t").diarize("a.wav")
    assert [(s.speaker, s.start_sec, s.end_sec) for s in result.segments] == [
        ("SPEAKER_00", 0.0, 1.23), ("SPEAKER_01", 1.5, 3.0), ("SPEAKER_00", 3.0, 4.0)]
    assert result.num_speakers == 2


def test_token_keyword_on_4x(monkeypatch):
    calls = _install_fake(monkeypatch, wrap_output=True, accepts_token=True)
    backend.PyannoteDiarizer(auth_token="secret").diarize("a.wav")
    assert calls["auth_kw"] == {"token": "secret"}


def test_falls_back_to_use_auth_token_on_3x(monkeypatch):
    calls = _install_fake(monkeypatch, wrap_output=False, accepts_token=False)
    backend.PyannoteDiarizer(auth_token="secret", model=backend.LEGACY_MODEL).diarize("a.wav")
    assert calls["auth_kw"] == {"use_auth_token": "secret"}
    assert calls["model"] == backend.LEGACY_MODEL


def test_no_hints_means_none_forwarded(monkeypatch):
    calls = _install_fake(monkeypatch, wrap_output=True, accepts_token=True)
    backend.PyannoteDiarizer(auth_token="t").diarize("a.wav")
    assert calls["hints"] == {}


def test_range_hints_are_forwarded(monkeypatch):
    calls = _install_fake(monkeypatch, wrap_output=True, accepts_token=True)
    backend.PyannoteDiarizer(auth_token="t").diarize("a.wav", min_speakers=9, max_speakers=12)
    assert calls["hints"] == {"min_speakers": 9, "max_speakers": 12}


def test_waveform_dict_passes_through_untouched(monkeypatch):
    calls = _install_fake(monkeypatch, wrap_output=True, accepts_token=True)
    audio = {"waveform": object(), "sample_rate": 16000}
    backend.PyannoteDiarizer(auth_token="t").diarize(audio, num_speakers=10)
    assert calls["audio"] is audio
    assert calls["hints"] == {"num_speakers": 10}
