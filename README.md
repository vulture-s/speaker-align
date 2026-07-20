# speaker-align

Attach speaker labels to transcript segments by temporal overlap.

Diarization tells you *when* each person spoke. ASR tells you *what* was said and
when. Neither tells you **who said which line** — that join is this package.

```python
from speaker_align import get_diarizer, align_speakers_to_transcript

turns = get_diarizer("pyannote", auth_token=TOKEN).diarize("meeting.wav")
segments = align_speakers_to_transcript(turns.segments, whisper_segments)
# [{"start": 0.0, "end": 4.2, "text": "...", "speaker": "SPEAKER_00"}, ...]
```

## Two halves, independently usable

| | needs | use it when |
|---|---|---|
| `align_speakers_to_transcript` | nothing but stdlib | you already have speaker turns |
| `get_diarizer(...)` | `pip install 'speaker-align[pyannote]'` | you need to produce them from audio |

Importing the alignment half never imports a backend, so a caller that gets
speaker turns elsewhere — a vendor API, a different diarizer, a human pass —
installs no torch and no pyannote. That split is enforced by a test.

## Install

```bash
pip install speaker-align                  # alignment only
pip install 'speaker-align[pyannote]'      # + local diarization
```

pyannote needs a Hugging Face token for
[speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).
Pass it as `auth_token=`, or set one of `SPEAKER_ALIGN_AUTH_TOKEN`,
`PYANNOTE_AUTH_TOKEN`, `HUGGINGFACE_TOKEN`. Prefer passing it explicitly when
embedding this in an application — it keeps that application's config in one
place instead of splitting it between a config object and ambient environment.

## How a label is decided

Each transcript segment goes to whichever speaker turn overlaps it most, but only
when the winning overlap is convincing:

* at least `tolerance` seconds (default `0.5`), **or**
* any overlap at all, when the segment is shorter than `tolerance * 2`

The second rule matters: a 0.3s interjection can never accumulate 0.5s of
overlap, so judging it by the same absolute bar would leave every short segment
unlabelled.

When neither holds, the segment gets `UNKNOWN_SPEAKER` (`""`) instead of a guess.
**A wrong attribution is worse than an absent one** — downstream, a multicam
editor may cut to the wrong camera because of it, and a missing label is visible
where a confident wrong one is not.

## API

```python
align_speakers_to_transcript(speaker_segments, transcript_segments, tolerance=0.5) -> list[dict]
align_segments_json(speaker_segments, transcript_segments_json, tolerance=0.5) -> str
speaker_turn_count(aligned_segments) -> int
get_diarizer(backend="pyannote", auth_token=None) -> BaseDiarizer
```

Transcript segments are plain dicts with `start` / `end` in seconds; every other
key is carried through untouched. Inputs are never mutated — you get copies.

`align_segments_json` is for callers that persist segments as a JSON column.
`speaker_turn_count` is a cheap downstream signal: a high count means dialogue
(cut between angles), a low one means monologue (hold, or cut to b-roll).

## Adding a backend

Subclass `BaseDiarizer`, return a `DiarizationResult`, and register it in
`get_diarizer`. The alignment half neither knows nor cares which backend ran.

## Origin

Extracted from [reel-scout](https://github.com/vulture-s)'s diarization module so
that other tools can share it. Behaviour is identical to the original: the
extraction was verified against it across 2,018 generated cases plus boundary
inputs (zero-length segments, segments straddling a turn boundary, missing
timing keys, empty inputs) with no divergence.

Sibling package: [whisper-guard](https://github.com/vulture-s/whisper-guard),
which filters hallucinations out of the transcript this one labels.

## License

MIT
