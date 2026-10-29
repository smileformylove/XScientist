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
- [x] Content-bound timed text indexes and offline interval retrieval.
- [x] Read-only inspection plans, exact frame selection and bounded execution.
- [x] Provider-input receipts and observed execution feedback.

Git stores source/evidence JSON and CAS pointers. Original videos and extracted
PNGs remain in the configured CAS and are excluded from source-code commits.
Identical bytes share CAS objects, including frames reused by observations.
The default source title contains no original filename or host path.

## Durable action receipts

Video search, indexing, inspection, and experiment orchestration can share the
append-only action ledger in `xscientist.research-action.v1`. An action has a
stable logical ID and records only hashes plus Research Object references.
Repeating a pending or terminal action with the same input hash is idempotent;
changing the input under the same ID is rejected. A worker crash can be
represented as `interrupted` and resumed explicitly, preserving attempt and
recovery lineage.

```python
from xscientist import ResearchActionLedger, ResearchRepository

repo = ResearchRepository("./my-study")
ledger = ResearchActionLedger(repo)
ledger.begin(
    "video-search-run-01",
    "video.search",
    {"query": "reaches marker", "scope": "registered-assets"},
    consumes=["rso-VIDEO_ASSET_ID"],
    budget_seconds=60,
)
ledger.finish(
    "video-search-run-01",
    "completed",
    {"evidence_ids": ["rso-VIDEO_EVIDENCE_ID"]},
    produces=["rso-VIDEO_EVIDENCE_ID"],
)
```

The receipt does not contain the private query, model prompt, raw media, or
local path. Bind assets, text indexes, observations, and experiment results
through `consumes`/`produces` relations and content hashes. The same API is
available through `xscientist research action begin|finish|list`.

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

## Timed Text and Retrieval

External transcription, OCR, captioning and shot-labeling tools can supply a
normalized JSON file. Times must already use the selected video's first decoded
frame as their origin; import rejects a missing or different origin.

```json
{
  "time_origin": "first_decoded_video_frame",
  "segments": [
    {"start_seconds": 1.0, "end_seconds": 2.5, "text": "Object reaches the marker"}
  ]
}
```

```bash
xscientist research video index-text segments.json --repo ./study \
  --asset rso-SOURCE_ID --kind transcript \
  --producer whisper --producer-version YOUR_VERSION --json
xscientist research video search "reaches marker" --repo ./study \
  --asset rso-SOURCE_ID --start 0 --end 10 --json
xscientist research video search "object motion" --repo ./study \
  --target rso-HYPOTHESIS_ID --limit 8 --json
```

Kinds are `transcript`, `ocr`, `caption` and `shot`. Indexes preserve the original
video hash, stream, tool identity/version, text hash and exact segment boundaries.
Segment text stays in CAS; Git stores a small immutable index. Inputs are limited
to 2 MiB, 4096 ordered segments and 4000 characters per segment. Overlapping
segments are allowed. Import runs no transcription/OCR model.

Retrieval uses the existing deterministic lexical-overlap helper, including CJK
features. It searches annotations and timed text, supports half-open time filters,
and returns source locators, text commitments, omission counts and verification
status. A targeted search reserves results for both explicit support and
refutation when present; its minimum limit is two. Text matches are retrieval
cues, not proof that a model inspected video or that a scientific claim holds.
Search runs without FFmpeg or a model provider and makes no graph changes.

## Planning, Inspection and Feedback

```bash
xscientist research video plan "object reaches the marker" --repo ./study \
  --asset rso-SOURCE_ID --max-frames 8 --max-decodes 2 --json
xscientist research video run "object reaches the marker" --repo ./study \
  --asset rso-SOURCE_ID --model YOUR_CONFIGURED_VLM \
  --max-frames 8 --max-decodes 2 --json
xscientist research video inspect --repo ./study --evidence rso-EVIDENCE_ID \
  --message "Describe only directly visible changes" --model YOUR_CONFIGURED_VLM \
  --max-frames 4 --json
xscientist research video feedback --repo ./study --asset rso-SOURCE_ID --json
```

`plan` is read-only and works without FFmpeg or provider dependencies. It
retrieves relevant windows, merges overlaps and checks the density of available
decoded points. Existing inputs are preferred when they meet that density;
missing or sparse intervals receive sampling actions. Frame and decode budgets
are explicit, temporal pairs are preferred, and omitted windows are counted.
Available points and points submitted during a prior successful provider call
are separate fields. Neither implies continuous coverage or model comprehension.

`run` executes the bounded plan, records sampled inputs without asserting a
scientific conclusion, and issues one logical VLM request through the existing
provider routing, budget and retry policy. `inspect` directly uses existing
evidence. These two commands call the configured model and may incur provider
costs; the other video commands do not. Default checkpoints remain exact-path,
and both commands accept `--no-commit`. Successfully prepared inputs remain
reusable if a later provider operation fails.

Existing-input plan actions carry exact `frame_refs` containing an evidence ID,
actual timestamp and frame hash. Pass those to `inspect_video_evidence()` or
`build_video_evidence_pack()` to preserve the focus window. CLI `context` and
`inspect` accept the same JSON list through `--frames-file`. Missing or altered
references fail before a provider call. Context budgets can still omit frames;
the actual input manifest and receipt report what was sent.

```python
from xscientist.video_planning import run_video_inspection

result = run_video_inspection(
    "./study", "rso-SOURCE_ID", "What changes near the marker?",
    client=client, model=model, max_frames=8, max_decodes=2,
)
response = result["response"]
receipt = result["inspection"]
```

An inspection receipt is a native `context_snapshot`, linked to the exact
evidence objects and retaining all referenced original/frame CAS objects.
It checks actual request image bytes against the input manifest and records
frame times/hashes, query/system/request/response digests, model route, observed
inspection latency and available ARA provider receipt references. Native Git
records contain no raw prompt, response or base64 image. The response is returned
to the caller; it is not automatically admitted as scientific evidence. Without
ARA tracing, `trace_status` explicitly remains `unavailable`. Failed provider
validation never creates a successful inspection receipt.

`feedback` validates receipt-to-evidence bindings and reports observed image
submissions, distinct source points, repeated points for identical query/system/
model combinations, latency and preparation counters from `run`. Asset filters
select receipts involving those assets, including joint multi-asset calls.
Missing preparation accounting and provider cost remain explicitly unknown.
Repeated submissions can be intentional. These host-recorded events provide
inputs for controlled Harness comparisons, not automatic skill adoption or
evidence that a policy improves research quality.

## Harness Improvement Roadmap

The [Harness2](https://github.com/google-research/harness2) reference is its
Propose-Probe-Compose loop over tool use, context, memory and procedures.
This implementation supplies actual assets and evidence inputs to that loop.
It does not claim to reproduce Harness2 or automate its improvement stages.

Remaining system changes, in order:

1. Asset reuse: exact frame derivations are reusable. Cache full timestamp
   indexes, add proxy and
   clip derivation receipts; recover missing objects through existing storage
   adapters without changing source identity.
2. Evidence retrieval: normalized shot/transcript/OCR/caption text and opposing
   observations are searchable. Add automatic extractor adapters and visual
   embedding retrieval with independently measured localization quality.
3. Adaptive inspection: density-based plans and bounded execution are available.
   Add model-requested follow-up windows, ROI inspection and provider cost
   resolution without overstating sparse point coverage.
4. Research orchestration: integrate video tool selection into the broader
   autonomous research loop and retain indexed/retrieved/submitted states.
5. Harness learning: use the recorded real preparation and provider-input events
   to probe bounded procedural changes with the existing refinement runtime,
   and admit scoped experience only after independent backtests.

Evaluate changes on tool selection, evidence localization, redundant decoding,
unsupported visual claims and total execution cost. Fixed tasks serve as
system regression inputs; they are separate from running a new video study.
