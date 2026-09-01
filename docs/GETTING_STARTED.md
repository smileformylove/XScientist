# Getting started

This guide follows the published `0.1.4` release. You need Python 3.10+ and Git.

## 1. Install, save your idea, and inspect it

No model, API key, Docker installation, or provider configuration is needed:

```bash
python -m pip install "xscientist==0.1.4"
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
python -m pip install "xscientist[research,openai]==0.1.4"
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
