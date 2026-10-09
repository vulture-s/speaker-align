# Changelog

## Unreleased

### Fixed
- **Wrong speaker when the real speaker's turn was fragmented.** A segment was
  given to the single turn with the largest overlap, so a speaker who said
  3.5s of a 5s line in three diarizer fragments (1.2s each) lost to someone
  who interjected for 1.3s at the end. Overlap is now summed per speaker
  (overlapping turns of one speaker unioned, not double-counted), and the
  `tolerance` bar is judged on that total.

## 0.1.0 — 2026-07-20

Initial release. Extracted from reel-scout's `reel_scout/diarize/` so arkiv,
reel-scout, and the meeting-transcribe toolchain can share one implementation
instead of each growing their own.

**Changes made during extraction:**

- Dropped the `from .. import config` coupling — `get_diarizer()` now takes
  `auth_token` explicitly, with env-var fallback for standalone use.
- The alignment primitive now works on `list[dict]` rather than only a JSON
  string; `align_segments_json()` keeps the JSON-in/JSON-out shape for callers
  that persist segments as a column.
- Inputs are no longer mutated — callers get copies.
- Backend import is lazy, so installing without `[pyannote]` gives a working
  alignment half and no torch. Enforced by a test.
- Added `speaker_turn_count()` for downstream editing decisions.
- `UNKNOWN_SPEAKER` is now a named constant rather than a bare `""`.

**Alignment behaviour is unchanged** — verified against the reel-scout original
across 2,018 generated cases plus boundary inputs, with no divergence.
