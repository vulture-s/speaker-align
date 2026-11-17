"""pyannote.audio backend.

Imported lazily by :func:`speaker_align.get_diarizer` so that installing this
package does not drag in pyannote/torch for callers that only want the alignment
half.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseDiarizer, DiarizationResult, SpeakerSegment

#: Default pipeline. Changed from ``speaker-diarization-3.1`` in 0.2.0: on a
#: 69-minute, 10-speaker single-mic meeting, 3.1 left to pick the count found 5
#: clusters with one holding 56% of all talk time (several people merged);
#: community-1 found 7, and with the count given, 32 of 40 hand-checked samples
#: were one person matching their cluster's majority. Needs pyannote.audio >= 4.
MODEL = "pyannote/speaker-diarization-community-1"
LEGACY_MODEL = "pyannote/speaker-diarization-3.1"


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
        # pyannote.audio 4 renamed ``use_auth_token`` to ``token``; 3.x only
        # knows the old name. Try the new one first so 4.x does not warn.
        try:
            self._pipeline = Pipeline.from_pretrained(
                self._model, token=self._auth_token
            )
        except TypeError:
            self._pipeline = Pipeline.from_pretrained(
                self._model, use_auth_token=self._auth_token
            )

    def diarize(
        self,
        audio_path: Any,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """Diarize ``audio_path``.

        ``audio_path`` is passed straight to pyannote, so it may also be a
        ``{"waveform": tensor, "sample_rate": int}`` dict -- the way around
        pyannote 4's torchcodec decoder when its FFmpeg libraries will not load
        (seen on Windows).

        Speaker-count hints are forwarded only when given. Known failure on a
        single far-field mic: the most talkative speaker can come back as two
        clusters. A ``min_speakers``/``max_speakers`` range does not fix that
        (tested 9-12 on a 10-person meeting: the model took 9 and still split
        that speaker), so pass the real count and merge the split pair when
        reviewing.
        """
        self._ensure_pipeline()
        hints = {}  # type: Dict[str, int]
        if num_speakers is not None:
            hints["num_speakers"] = num_speakers
        if min_speakers is not None:
            hints["min_speakers"] = min_speakers
        if max_speakers is not None:
            hints["max_speakers"] = max_speakers
        output = self._pipeline(audio_path, **hints)
        # pyannote 4 returns a DiarizeOutput wrapping the annotation; 3.x
        # returns the annotation itself.
        diarization = getattr(output, "speaker_diarization", output)

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
