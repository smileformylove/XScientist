# Getting started

This guide follows the published `0.1.5` release. You need Python 3.10+ and Git.

## 1. Install, save your idea, and inspect it

No model, API key, Docker installation, or provider configuration is needed:

```bash
python -m pip install "xscientist==0.1.5"
xscientist explore ./my-study
xscientist status ./my-study
```

`explore` asks for an expected observation, a result that would change your
mind, and a fair first test. Answer only what you know. `status` then shows one
highest-priority next action without inventing evidence or conclusions.

To inspect a complete provider-free example:

```bash
xscientist demo ./first-study --autopilot
xscientist status ./first-study
```

The demo costs `$0.00` and intentionally ends with “more evidence needed.” Its
held-out result challenges an over-broad claim; preserving that conflict is a
successful scientific outcome, not a software failure.

## 2. Optionally add a model

Install the research runtime plus exactly one provider client. This example uses
OpenAI; choose the matching provider extra for another service:

```bash
python -m pip install "xscientist[research,openai]==0.1.5"
export OPENAI_API_KEY="..."
xscientist start ./my-study --prepare-only
```

The question saved by `explore` is reused. `--prepare-only` creates or updates
the workspace and validates local prerequisites without starting the study.
Resolve the reported blocker before continuing.

When the workspace is ready, set an explicit budget and start the run:

```bash
xscientist start ./my-study --max-cost-usd 10
xscientist status ./my-study
```

A local Ollama model needs no hosted key, but generated experiment code still
requires the configured isolated executor. Provider presence and scientific
verification are separate states.

## 3. Publication-oriented workflow

Use the same safe preparation step first. Then select publication autopilot
with an explicit budget:

```bash
xscientist start ./my-study --prepare-only
xscientist start ./my-study --autopilot publication --max-cost-usd 10
xscientist status ./my-study
```

Publication autopilot organizes research, writeup, and review gates. It does
not promise manuscript completion, scientific verification, venue submission,
or acceptance.

## 4. Search sources and inspect video evidence

For web or video discovery, record a bounded search receipt first:

```bash
xscientist research web search "video evidence for intervention X" \
  --vertical video --provider auto --limit 20 --repo ./my-study
```

Select a candidate before treating it as a source. For local media, import the
video into the content-addressed store, create a timestamped observation, and
retrieve only the frames needed by the question:

```bash
xscientist research video add ./input.mp4 --repo ./my-study --title "Run A"
xscientist research video observe --repo ./my-study \
  --asset rso-SOURCE_ID --start 1 --end 3 \
  --result "Object motion changes" --max-frames 4
```

Import and extraction need FFmpeg/ffprobe. Original media stays in CAS rather
than source Git. See [Auditable Web Search](WEB_SEARCH.md) and
[Video Assets and Evidence](VIDEO_EVIDENCE.md) for provider configuration,
timed-text retrieval, VLM inspection budgets, and privacy boundaries.

## Continue only when needed

- [Long-running guide](LONG_RUNNING_GUIDE.md): detach, watch, cancel, and resume.
- [Local Research Git](LOCAL_RESEARCH_GIT.md): inspect, save, diff, and recover checkpoints.
- [Research integrity](RESEARCH_INTEGRITY.md): traceability and independent review boundaries.
- [Configuration reference](CONFIG_REFERENCE.md): providers, executors, and specialist capabilities.

The `main` branch also documents unreleased protocol work. Install it from
source only when you need a feature explicitly labeled **Development main**:

```bash
python -m pip install \
  "xscientist @ git+https://github.com/smileformylove/XScientist.git@main"
```

Pin a source commit instead of `main` for repeatable research.
