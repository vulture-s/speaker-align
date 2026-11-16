# Known bug classes

| Class | First instance (fixed in) | Single-case regression (`tests/test_align.py`) | Class-level test (`tests/test_align_invariants.py`) |
|---|---|---|---|
| The label depends on how the diarizer sliced the audio, not on who spoke (overlap judged per turn) | three 1.2 s fragments of A lost to one 1.3 s turn of B (#1) | `test_fragmented_speaker_beats_a_longer_single_turn`, `test_tolerance_is_judged_on_the_speakers_total_overlap`, `test_overlapping_turns_of_one_speaker_are_not_double_counted` | `test_fragmenting_turns_never_changes_a_label`, `test_duplicate_turns_never_change_a_label`, `test_turn_order_never_changes_a_label`, `test_label_is_the_speaker_with_most_total_overlap` (brute-force oracle), `test_never_labels_a_speaker_with_no_overlap` |

Rule of thumb: the alignment is a function of *who spoke when*. Any change to
`align.py` must keep it invariant under splitting, duplicating and reordering
speaker turns — the invariant tests sweep 300 random layouts for that.

A wrong label is worse than none (multicam editors cut to the wrong angle), so
the "unknown rather than a guess" tests in `test_align.py` stay authoritative.

There is no CI in this repo; `python -m pytest` from the repo root runs all of
the above.
