# Changelog

## 0.2.0 — 2026-11-17

### Changed
- **Default pipeline is now `pyannote/speaker-diarization-community-1`**
  (was `speaker-diarization-3.1`); the `[pyannote]` extra now needs
  `pyannote.audio>=4.0`. On a 69-minute, 10-person single-mic meeting, 3.1
  left to pick the count found 5 clusters, one holding 56% of the talk time
  (several people merged); community-1 found 7, and with the count given, 32
  of 40 hand-checked samples were one person matching their cluster's
  majority. 3.1 is still reachable as `LEGACY_MODEL`.

### Added
- `diarize()` takes `num_speakers` / `min_speakers` / `max_speakers` and
  forwards only the ones given.
- `audio_path` may be a `{"waveform", "sample_rate"}` dict, passed through to
  pyannote untouched — the way around pyannote 4's torchcodec decoder when its
  FFmpeg libraries will not load (seen on Windows).

### Fixed
- Works with pyannote.audio 4: `token=` instead of the removed
  `use_auth_token=` (falls back for 3.x), and the `DiarizeOutput` wrapper is
  unwrapped before reading tracks.

## 0.1.1 — 2026-11-16

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
