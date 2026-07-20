"""speaker-align — attach speaker labels to transcript segments.

Two halves, independently usable:

* **align** (:mod:`speaker_align.align`) — pure logic, zero heavy dependencies.
  Give it speaker turns and transcript segments, get segments tagged with who
  said them.
* **diarize** (:mod:`speaker_align.pyannote`) — produces those speaker turns from
  audio. Optional: ``pip install 'speaker-align[pyannote]'``.

Callers that already have speaker turns (from a vendor API, a different
diarizer, or a human pass) use the first half alone and never install pyannote.
"""
from __future__ import annotations

import os
from typing import Optional

from .align import (
    UNKNOWN_SPEAKER,
    align_segments_json,
    align_speakers_to_transcript,
    speaker_turn_count,
)
from .base import BaseDiarizer, DiarizationResult, SpeakerSegment

__version__ = "0.1.0"

__all__ = [
    "BaseDiarizer",
    "DiarizationResult",
    "SpeakerSegment",
    "UNKNOWN_SPEAKER",
    "align_segments_json",
    "align_speakers_to_transcript",
    "get_diarizer",
    "speaker_turn_count",
    "__version__",
]

#: Checked in order when ``auth_token`` is not passed explicitly.
AUTH_TOKEN_ENV_VARS = ("SPEAKER_ALIGN_AUTH_TOKEN", "PYANNOTE_AUTH_TOKEN",
                       "HUGGINGFACE_TOKEN")


def _token_from_env() -> str:
    for name in AUTH_TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def get_diarizer(
    backend: Optional[str] = None, auth_token: Optional[str] = None
) -> BaseDiarizer:
    """Build a diarizer for ``backend`` (default ``"pyannote"``).

    ``auth_token`` is taken from the argument when given, otherwise from the
    first set variable in :data:`AUTH_TOKEN_ENV_VARS`. Passing it explicitly is
    preferred for embedders — it keeps the host application's config in one
    place instead of splitting it between config objects and ambient env.
    """
    backend = backend or "pyannote"
    token = auth_token if auth_token is not None else _token_from_env()

    if backend == "pyannote":
        from .pyannote import PyannoteDiarizer

        return PyannoteDiarizer(auth_token=token)
    raise ValueError("Unknown diarization backend: %s" % backend)
