# -*- coding: utf-8 -*-
"""Invariants of the alignment, swept over random inputs.

#1 fixed one instance of a class: the answer depended on *how the diarizer
happened to slice* a speaker's speech (three 1.2s fragments lost to one 1.3s
turn). The label must depend only on who spoke when — not on fragmentation,
duplication or the order turns arrive in. These tests check that property on
hundreds of random layouts, so a future "optimisation" of the overlap loop
that reintroduces per-turn judging fails here. See docs/known-bug-classes.md.
"""
import random

from speaker_align import SpeakerSegment, align_speakers_to_transcript

SPEAKERS = ["A", "B", "C"]


def _layout(rng, length=60.0):
    """Non-overlapping turns covering [0, length) with gaps, random speakers."""
    turns, t = [], 0.0
    while t < length:
        dur = rng.uniform(0.2, 6.0)
        if rng.random() < 0.8:
            turns.append(SpeakerSegment(rng.choice(SPEAKERS), round(t, 3), round(t + dur, 3)))
        t += dur + rng.choice([0.0, 0.0, rng.uniform(0.05, 1.0)])
    return turns


def _transcript(rng, length=60.0):
    segs, t = [], 0.0
    while t < length:
        dur = rng.uniform(0.2, 8.0)
        segs.append({"start": round(t, 3), "end": round(t + dur, 3), "text": "x"})
        t += dur
    return segs


def _fragment(rng, turns):
    """Split each turn into 1-4 contiguous pieces (what pyannote does at pauses)."""
    out = []
    for turn in turns:
        cuts = sorted(rng.uniform(turn.start_sec, turn.end_sec)
                      for _ in range(rng.randrange(0, 4)))
        edges = [turn.start_sec] + cuts + [turn.end_sec]
        out += [SpeakerSegment(turn.speaker, a, b) for a, b in zip(edges, edges[1:]) if b > a]
    return out


def _labels(turns, transcript):
    return [s["speaker"] for s in align_speakers_to_transcript(turns, transcript)]


def _cases(n=300):
    rng = random.Random(20261009)
    for _ in range(n):
        yield rng, _layout(rng), _transcript(rng)


def test_fragmenting_turns_never_changes_a_label():
    for rng, turns, transcript in _cases():
        assert _labels(_fragment(rng, turns), transcript) == _labels(turns, transcript)


def test_duplicate_turns_never_change_a_label():
    for rng, turns, transcript in _cases():
        assert _labels(turns + turns, transcript) == _labels(turns, transcript)


def test_turn_order_never_changes_a_label():
    for rng, turns, transcript in _cases():
        shuffled = list(turns)
        rng.shuffle(shuffled)
        # Exact ties (equal summed overlap) are resolved by first-seen order;
        # random float layouts make those vanishingly rare, so skip if seen.
        assert _labels(shuffled, transcript) == _labels(turns, transcript)


def test_label_is_the_speaker_with_most_total_overlap():
    """Brute-force oracle: integrate overlap per speaker on a fine grid."""
    step = 0.01
    for rng, turns, transcript in _cases(n=60):
        out = align_speakers_to_transcript(turns, transcript)
        for seg, aligned in zip(transcript, out):
            if not aligned["speaker"]:
                continue
            totals = {}
            t = seg["start"] + step / 2
            while t < seg["end"]:
                for turn in turns:
                    if turn.start_sec <= t < turn.end_sec:
                        totals[turn.speaker] = totals.get(turn.speaker, 0) + step
                        break
                t += step
            best = max(totals.values())
            # within grid error of the true argmax
            assert totals.get(aligned["speaker"], 0) >= best - 2 * step - 1e-9, (seg, totals, aligned)


def test_never_labels_a_speaker_with_no_overlap():
    for rng, turns, transcript in _cases():
        for seg, aligned in zip(transcript, align_speakers_to_transcript(turns, transcript)):
            if aligned["speaker"]:
                assert any(t.speaker == aligned["speaker"]
                           and min(t.end_sec, seg["end"]) > max(t.start_sec, seg["start"])
                           for t in turns)
