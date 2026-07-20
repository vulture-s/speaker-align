"""Backend-agnostic diarization types.

The dataclasses here are the package's public contract: a diarizer returns
``DiarizationResult``, and ``align_speakers_to_transcript`` consumes
``SpeakerSegment`` values. Keeping this module free of any backend import is what
lets a caller depend on the alignment half without pulling in pyannote (or torch)
at all — see README "Two halves, independently usable".
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List


@dataclass
class SpeakerSegment:
    """One contiguous stretch of audio attributed to a single speaker."""

    speaker: str  # backend label, e.g. "SPEAKER_00"
    start_sec: float
    end_sec: float


@dataclass
class DiarizationResult:
    segments: List[SpeakerSegment] = field(default_factory=list)
    num_speakers: int = 0


class BaseDiarizer(abc.ABC):
    """Implement this to add a backend; see ``pyannote.PyannoteDiarizer``."""

    @abc.abstractmethod
    def diarize(self, audio_path: str) -> DiarizationResult:
        ...
