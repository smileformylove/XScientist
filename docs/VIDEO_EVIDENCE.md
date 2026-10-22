# Video Assets and Evidence

This is a system capability: videos become reusable, content-addressed sources
and time-localized inputs to research decisions. It does not launch a video
research campaign or certify an observation as a scientific result.

## Available

- [x] Immutable original video storage in the existing Research VCS CAS.
- [x] Native source indexes with hashes, duration, dimensions, container,
  video stream identity, audio presence and probe version.
- [x] Interval observations with actual decoded frame timestamps and provenance.
- [x] Bounded evidence packs, temporal locators in research context and VLM input.
- [x] Native checkpoint retention, integrity checks and regression tests.
- [x] Verified frame reuse, temporal frame allocation and duplicate image removal.

Git stores source/evidence JSON and CAS pointers. Original videos and extracted
PNGs remain in the configured CAS and are excluded from source-code commits.
Identical bytes share CAS objects, including frames reused by observations.
The default source title contains no original filename or host path.

## Native Commands

FFmpeg and ffprobe must be installed on the machine performing import and
extraction. Listing sources and compiling existing evidence need neither tool.
Use an existing initialized research repository:

```bash
xscientist research video add /path/to/input.mp4 --repo ./study --title "Run A" --json
xscientist research video list --repo ./study --json
xscientist research video observe --repo ./study \
  --asset rso-SOURCE_ID --start 1.0 --end 3.0 \
  --result "Object motion changes after the intervention" --max-frames 4 --json
xscientist research video context --repo ./study \
  --evidence rso-EVIDENCE_ID --max-frames 4 --budget-tokens 2000 --json
```

The two recording commands create exact-path native checkpoints by default;
`--no-commit` leaves the records pending. `observe` also accepts explicit
`--supports` and `--refutes` graph targets. Observations stay completed,
unverified records; none of these operations promotes a scientific claim.
Missing tools, invalid intervals, unsafe source links and corrupt CAS content
are rejected. Failed publication rolls back new pointers and typed records
when no checkpoint has been published; unreferenced CAS bytes can remain.

`observe` reuses an existing derivation when the original content, stream,
interval and sampling targets match. Original and frame hashes are verified
before reuse; the original decoder version stays attached to the derivation.
Reuse also works without FFmpeg. `--refresh` explicitly requests new decoding.
JSON results include observed cache-hit, decoded-frame and reused-frame counts;
these operation counters are separate from scientific observation content.

## Time and Sampling

Intervals are half-open `[start, end)` seconds relative to the first decoded
frame of the selected video stream. Attached cover pictures are excluded.
Duration comes from container/stream metadata and is normalized where needed;
it is not a guarantee that every time in the interval has a decodable frame.
Extraction uses FFmpeg's timestamped frames rather than average FPS arithmetic.
Frame times are computed from integer presentation timestamps and the filter
time base. Nonzero starts and variable frame rates are covered by integration
tests. Stream changes or ambiguous decoder timestamps fail closed.

Uniform target times select the first available frame at or after each target.
Several targets can map to one frame, so fewer frames may be returned. Frames
are resized to at most 768 pixels per dimension. The evidence records the
source hash, stream, interval, targets, actual frame timestamps, frame hashes
and decoder version. Current limits are 16 GiB per original, 16 frames per
observation, 4 MiB per PNG, 30 seconds for probing and 120 seconds for extraction.
Extraction decodes from the beginning to preserve normalized timestamps; a
late interval in a long video can hit that timeout.

Media tools run as bounded local subprocesses with local file/pipe protocols.
This is not an OS sandbox. Process untrusted media inside the configured
isolated environment, just as with other external research tools.

## Context and VLM Integration

Native research context follows the existing support/refutation graph. With
explicit evidence objectives, localized views carry a content-bound
`video:sha256:...#t=start,end` locator. These text views identify evidence;
they do not mean a model has inspected the corresponding images.

`build_video_evidence_pack()` validates source bindings and CAS hashes and
returns a bounded metadata view. Its frame budget is global across all supplied
evidence IDs. Allocation proceeds in rounds across evidence and favors temporal
spread within an interval. Identical source/stream/timestamp/frame references
share one image attachment without collapsing their distinct annotations.
Omitted frames/evidence are counted; the token budget covers the
metadata pack, not image tokens or the caller's message. Originals shared by
several observations are hashed once per compilation.

The existing VLM entry point accepts timestamped video evidence while keeping
its provider routing, response validation, token budget and call receipts:

```python
from ai_scientist.vlm import get_response_from_vlm

response, history = get_response_from_vlm(
    "Compare these observations and cite their evidence IDs and timestamps.",
    [], client, model, system_message,
    video_repo="./study",
    video_evidence_ids=["rso-EVIDENCE_ID"],
    max_images=4,
)
```

`prepare_video_evidence_prompt()` builds the same timestamp/image content
without a model call. Audio and unsampled intervals are explicitly outside
this visual evidence. User annotations remain unverified, and media text is
source data rather than executable instructions.

## Harness Improvement Roadmap

The [Harness2](https://github.com/google-research/harness2) reference is its
Propose-Probe-Compose loop over tool use, context, memory and procedures.
This implementation supplies actual assets and evidence inputs to that loop.
It does not claim to reproduce Harness2 or automate its improvement stages.

Next system changes, in order:

1. Asset reuse: cache timestamp indexes and frame derivatives; add proxy and
   clip derivation receipts; recover missing objects through existing storage
   adapters without changing source identity.
2. Evidence retrieval: combine shot, transcript, OCR and visual indexes;
   retrieve relevant intervals and retain contradictory observations.
3. Adaptive inspection: use coarse-to-fine sampling and request more frames
   near uncertainty, always recording actual inspected coverage and cost.
4. Research orchestration: select these tools from task/context requirements,
   preserving a distinction between indexed, retrieved and model-inspected
   evidence instead of equating metadata access with visual inspection.
5. Harness learning: instrument real asset resolution and inspection events,
   probe bounded procedural changes with the existing refinement runtime,
   and admit scoped experience only after independent backtests.

Evaluate changes on tool selection, evidence localization, redundant decoding,
unsupported visual claims and total execution cost. Fixed tasks serve as
system regression inputs; they are separate from running a new video study.
