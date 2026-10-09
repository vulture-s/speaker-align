# -*- coding: utf-8 -*-
import json

import pytest

from speaker_align import (
    UNKNOWN_SPEAKER,
    SpeakerSegment,
    align_segments_json,
    align_speakers_to_transcript,
    speaker_turn_count,
)


def seg(start, end, text="x"):
    return {"start": start, "end": end, "text": text}


TWO_SPEAKERS = [
    SpeakerSegment("SPEAKER_00", 0.0, 10.0),
    SpeakerSegment("SPEAKER_01", 10.0, 20.0),
]


def test_assigns_speaker_by_maximum_overlap():
    out = align_speakers_to_transcript(TWO_SPEAKERS, [seg(1.0, 4.0), seg(12.0, 15.0)])
    assert [s["speaker"] for s in out] == ["SPEAKER_00", "SPEAKER_01"]


def test_straddling_segment_goes_to_the_larger_overlap():
    # 8.0-13.0: 2s of SPEAKER_00, 3s of SPEAKER_01 -> the latter wins.
    out = align_speakers_to_transcript(TWO_SPEAKERS, [seg(8.0, 13.0)])
    assert out[0]["speaker"] == "SPEAKER_01"


def test_no_overlap_yields_unknown_not_a_guess():
    out = align_speakers_to_transcript(TWO_SPEAKERS, [seg(50.0, 55.0)])
    assert out[0]["speaker"] == UNKNOWN_SPEAKER


def test_thin_overlap_on_a_long_segment_is_rejected():
    # 0.2s of overlap on a 5s segment: below tolerance, and the segment is long
    # enough that it should have accumulated more -> refuse to label it.
    turns = [SpeakerSegment("SPEAKER_00", 9.8, 20.0)]
    out = align_speakers_to_transcript(turns, [seg(5.0, 10.0)], tolerance=0.5)
    assert out[0]["speaker"] == UNKNOWN_SPEAKER


def test_short_segment_keeps_label_despite_sub_tolerance_overlap():
    # A 0.3s interjection can never reach 0.5s of overlap; judging it by the
    # absolute bar would leave every short segment unlabelled.
    turns = [SpeakerSegment("SPEAKER_01", 0.0, 20.0)]
    out = align_speakers_to_transcript(turns, [seg(1.0, 1.3)], tolerance=0.5)
    assert out[0]["speaker"] == "SPEAKER_01"


def test_inputs_are_not_mutated():
    original = seg(1.0, 4.0)
    snapshot = dict(original)
    align_speakers_to_transcript(TWO_SPEAKERS, [original])
    assert original == snapshot, "caller's segments must not gain a speaker key"


def test_other_segment_fields_survive():
    out = align_speakers_to_transcript(
        TWO_SPEAKERS, [{"start": 1.0, "end": 2.0, "text": "嗨", "confidence": 0.9}]
    )
    assert out[0]["text"] == "嗨"
    assert out[0]["confidence"] == 0.9


def test_empty_transcript_and_empty_turns():
    assert align_speakers_to_transcript(TWO_SPEAKERS, []) == []
    out = align_speakers_to_transcript([], [seg(1.0, 2.0)])
    assert out[0]["speaker"] == UNKNOWN_SPEAKER


def test_missing_start_end_defaults_to_zero_and_does_not_raise():
    out = align_speakers_to_transcript(TWO_SPEAKERS, [{"text": "no timing"}])
    assert out[0]["speaker"] == UNKNOWN_SPEAKER


def test_json_wrapper_round_trips_and_preserves_unicode():
    payload = json.dumps([{"start": 1.0, "end": 4.0, "text": "這批線材"}])
    result = json.loads(align_segments_json(TWO_SPEAKERS, payload))
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["text"] == "這批線材"


@pytest.mark.parametrize(
    "speakers,expected",
    [
        ([], 0),
        (["A"], 0),
        (["A", "A", "A"], 0),
        (["A", "B"], 1),
        (["A", "B", "A"], 2),
    ],
)
def test_speaker_turn_count(speakers, expected):
    segments = [{"speaker": s} for s in speakers]
    assert speaker_turn_count(segments) == expected


def test_turn_count_skips_unknown_instead_of_counting_it_as_a_change():
    # One unlabelled line inside a single answer must not read as two turns.
    segments = [{"speaker": "A"}, {"speaker": UNKNOWN_SPEAKER}, {"speaker": "A"}]
    assert speaker_turn_count(segments) == 0


def test_fragmented_speaker_beats_a_longer_single_turn():
    # pyannote splits one person's speech at short pauses. A speaks 3.5s of
    # this 5s line in three turns (1.2 + 1.2 + 1.1); B speaks 1.3s at the end.
    # Judged turn-by-turn, B's single 1.3s turn "wins" -> wrong camera.
    turns = [
        SpeakerSegment("A", 0.0, 1.2),
        SpeakerSegment("A", 1.3, 2.5),
        SpeakerSegment("A", 2.6, 3.7),
        SpeakerSegment("B", 3.7, 5.0),
    ]
    out = align_speakers_to_transcript(turns, [seg(0.0, 5.0)])
    assert out[0]["speaker"] == "A"


def test_tolerance_is_judged_on_the_speakers_total_overlap():
    # Two 0.3s fragments of the same speaker on a long segment: neither alone
    # reaches 0.5s, together they do.
    turns = [SpeakerSegment("A", 1.0, 1.3), SpeakerSegment("A", 2.0, 2.3)]
    out = align_speakers_to_transcript(turns, [seg(0.0, 5.0)], tolerance=0.5)
    assert out[0]["speaker"] == "A"


def test_overlapping_turns_of_one_speaker_are_not_double_counted():
    # Duplicate/overlapping turns for A must count once (0-2s = 2s), so B's
    # 3s still wins.
    turns = [
        SpeakerSegment("A", 0.0, 2.0),
        SpeakerSegment("A", 0.0, 2.0),
        SpeakerSegment("B", 2.0, 5.0),
    ]
    out = align_speakers_to_transcript(turns, [seg(0.0, 5.0)])
    assert out[0]["speaker"] == "B"
