"""pyannote.audio backend.

Imported lazily by :func:`speaker_align.get_diarizer` so that installing this
package does not drag in pyannote/torch for callers that only want the alignment
half.
"""
from __future__ import annotations

from typing import Optional

from .base import BaseDiarizer, DiarizationResult, SpeakerSegment

MODEL = "pyannote/speaker-diarization-3.1"


class PyannoteDiarizer(BaseDiarizer):
    def __init__(self, auth_token: str = "", model: str = MODEL) -> None:
        self._auth_token = auth_token
        self._model = model
        self._pipeline = None  # type: Optional[object]

    def _ensure_pipeline(self) -> None:
        """Load the pipeline once, on first use.

        Deferred rather than done in ``__init__`` because loading pulls model
        weights over the network; constructing a diarizer stays cheap and
        side-effect free, which is what makes it safe to build one up front and
        only pay for it if diarization is actually requested.
        """
        if self._pipeline is not None:
            return
        try:
            from pyannote.audio import Pipeline  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "pyannote.audio not installed. Install with: "
                "pip install 'speaker-align[pyannote]'"
            )
        if not self._auth_token:
            raise ValueError(
                "auth_token is required for %s. Get one from "
                "https://huggingface.co/%s" % (self._model, self._model)
            )
        self._pipeline = Pipeline.from_pretrained(
            self._model, use_auth_token=self._auth_token
        )

    def diarize(self, audio_path: str) -> DiarizationResult:
        self._ensure_pipeline()
        diarization = self._pipeline(audio_path)

        segments = []
        speakers = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                SpeakerSegment(
                    speaker=speaker,
                    start_sec=round(turn.start, 2),
                    end_sec=round(turn.end, 2),
                )
            )
            speakers.add(speaker)

        return DiarizationResult(segments=segments, num_speakers=len(speakers))
