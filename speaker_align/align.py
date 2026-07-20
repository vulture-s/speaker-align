"""Attach speaker labels to transcript segments by temporal overlap.

This is the half of the package that carries the actual logic. It has no
dependency on any diarization backend — hand it speaker turns from anywhere
(pyannote, a vendor API, a hand-written list) and transcript segments from any
ASR, and it decides who said each line.

The primitive operates on plain ``list[dict]`` so it composes with anything.
``align_segments_json`` is a thin convenience wrapper for callers that keep
transcript segments as a JSON string in a database column.
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence

from .base import SpeakerSegment

#: Assigned to segments that no speaker turn overlaps convincingly.
UNKNOWN_SPEAKER = ""


def align_speakers_to_transcript(
    speaker_segments: Sequence[SpeakerSegment],
    transcript_segments: Sequence[Dict],
    tolerance: float = 0.5,
) -> List[Dict]:
    """Return a copy of ``transcript_segments`` with a ``speaker`` key added.

    Each transcript segment is attributed to whichever speaker turn overlaps it
    most. The label is only assigned when the winning overlap is convincing:

      * at least ``tolerance`` seconds of overlap, OR
      * any overlap at all, when the segment is shorter than ``tolerance * 2``
        — a 0.3s interjection can never accumulate 0.5s of overlap, so judging it
        by the same absolute bar would leave every short segment unlabelled.

    Otherwise the segment gets :data:`UNKNOWN_SPEAKER` rather than a guess:
    a wrong attribution is worse than an absent one downstream, where an editor
    may cut to the wrong camera because of it.

    Inputs are never mutated; each returned dict is a shallow copy.
    """
    aligned = []  # type: List[Dict]

    for seg in transcript_segments:
        out = dict(seg)
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))

        best_speaker = UNKNOWN_SPEAKER
        best_overlap = 0.0
        for turn in speaker_segments:
            overlap = min(seg_end, turn.end_sec) - max(seg_start, turn.start_sec)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker

        is_short = (seg_end - seg_start) < tolerance * 2
        if best_overlap >= tolerance or (best_overlap > 0 and is_short):
            out["speaker"] = best_speaker
        else:
            out["speaker"] = UNKNOWN_SPEAKER
        aligned.append(out)

    return aligned


def align_segments_json(
    speaker_segments: Sequence[SpeakerSegment],
    transcript_segments_json: str,
    tolerance: float = 0.5,
) -> str:
    """JSON-string in, JSON-string out wrapper around the primitive above.

    For callers that persist transcript segments as a JSON column and want to
    round-trip without unpacking (reel-scout's ``transcripts.segments_json``).
    """
    segments = json.loads(transcript_segments_json)
    aligned = align_speakers_to_transcript(speaker_segments, segments, tolerance)
    return json.dumps(aligned, ensure_ascii=False)


def speaker_turn_count(aligned_segments: Sequence[Dict]) -> int:
    """How many times the speaker changes across an aligned transcript.

    Cheap signal for downstream editing: a high turn count means dialogue
    (cut between angles), a low one means monologue (hold, or cut to b-roll).
    Unknown-speaker segments are skipped rather than counted as a change, so a
    single unlabelled line inside one person's answer does not read as two turns.
    """
    turns = 0
    previous = None
    for seg in aligned_segments:
        speaker = seg.get("speaker") or UNKNOWN_SPEAKER
        if speaker == UNKNOWN_SPEAKER:
            continue
        if previous is not None and speaker != previous:
            turns += 1
        previous = speaker
    return turns
