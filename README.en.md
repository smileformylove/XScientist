# XScientist

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Smoke](https://github.com/smileformylove/ai_scientist/actions/workflows/smoke.yml/badge.svg)](https://github.com/smileformylove/ai_scientist/actions/workflows/smoke.yml)

Chinese README: [README.md](README.md)

> A sustainable, self-improving autonomous research system: idea generation, experiment execution, paper writing, self-review loops, strategy scheduling, and long-running daemon ops.

XScientist is not built to "generate one paper once". It is designed as an operational research pipeline that can run continuously, stay observable, and produce handoff-ready artifacts (plans, evidence, reviews, repair tasks, quality gates, and reports) for iterative improvement and collaboration.

Important notes:

- Cost: running the system calls LLMs / retrieval services and may incur API fees and long runtimes.
- Reliability: model outputs may contain errors or hallucinations; verify key claims, data, and citations yourself.
- Output isolation: by default, run outputs are written outside this git repo (to avoid polluting an open-source repository).

---

## Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Outputs & Observability](#outputs--observability)
- [Docs](#docs)
- [Development](#development)
- [Roadmap](#roadmap)
- [Contributing & Community](#contributing--community)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Overview

Think of XScientist as a "research operating system":

- Input: topics / sources / constraints (budget, stop conditions, quality gates)
- Process: ideation -> experiments -> writing -> self-review -> repair/rewrite -> packaging
- Output: reusable research assets (reports, paper drafts, review/repair queues, run index, handoff briefs)

Core loop (simplified):

```mermaid
flowchart LR
  A["Topic / Sources"] --> B["Ideation & Ranking"]
  B --> C["Experiments"]
  C --> D["Writeup & Quality Gates"]
  D --> E["Self-Review & Repair"]
  E --> F["Artifacts + Index + Dossier"]
  F --> G["Daemon Strategy Feedback"]
  G --> B
```

---

## Key Features

- Self-review loop: multi-round self-review produces structured issues + repair plans, and enforces regression/coverage gates.
- Measurable experiment TODO closure: turns "missing evidence" into explicit TODOs and tracks closure progress.
- Long-running daemon: continuous execution, failure protection, source scheduling, trend reports, handoff briefs, and strategy feedback.
- Observability and replay: critical stage artifacts are written as structured files (JSON/MD) for comparison and post-mortems.
- Engineering safeguards: login guard, preflight/repo validation, config schemas, output directory isolation.

Entrypoints:

- `run_project.py`: single-project end-to-end run (good for local debugging and reproducing a run)
- `continuous_paper_generator.py`: continuous/batch generation
- `continuous_research_daemon.py`: long-running autonomous scheduling
- `research_manager.py`: index + boards (filtering, exporting, packaging)

---

## Quick Start

### 0) Prerequisites

- Python: 3.10+ (3.11 recommended)
- System deps (recommended):
  - LaTeX toolchain (to compile paper PDFs, e.g., TeX Live / MacTeX)
  - `poppler` (PDF processing/extraction)
  - `chktex` (optional LaTeX lint)

> GPU/CUDA is optional. If you need GPU acceleration, install the matching PyTorch build following the official PyTorch instructions.

### 1) Install (conda recommended)

```bash
conda create -n xscientist python=3.11 -y
conda activate xscientist

pip install -r requirements.txt
```

More reproducible (CI-style) install (optional):

```bash
pip install -r requirements.txt -c constraints-ci.txt
```

### 2) Configure API keys (as needed)

Set the env vars for your provider(s) (you do not need all of them):

```bash
export OPENAI_API_KEY="..."
export ZHIPU_API_KEY="..."
export GEMINI_API_KEY="..."
export S2_API_KEY="..."
```

### 3) Login (required)

```bash
python3 auth_cli.py login --user <your_name>
python3 auth_cli.py status
```

Login guard doc: `docs/LOGIN_GUARDRAIL.md`

### 4) Preflight (recommended)

```bash
python3 preflight_check.py --strict
python3 validate_repo.py
make smoke
```

---

## Configuration

### Output directory (do not write into the repo by default)

To keep the repo clean, outputs are written to a sibling directory by default:

- Default output root: `../ai_scientist_outputs` (relative to this repo)
- Priority: `RESEARCH_OUTPUT_DIR` > `AI_SCIENTIST_OUTPUT_DIR` > default sibling dir
- Fallback: if the sibling dir is not writable, use a system data dir (e.g., `~/.local/share/ai_scientist/research`)

Recommended: set an explicit output root.

```bash
export RESEARCH_OUTPUT_DIR="/path/to/my_xscientist_outputs"
```

### Strict fallback policy (debugging note)

Most scripts support stricter quality gates. During local debugging you may choose to relax strict fallbacks via `--override-strict-fallbacks` (not recommended for serious runs).

---

## Usage

### A) Run a single project from a topic

```bash
python3 run_project.py my_project \
  --output-root "$RESEARCH_OUTPUT_DIR" \
  --topic examples/example_topic.md
```

More usage: `docs/guides/PROJECT_USAGE.md`

### B) Continuous/batch generation

```bash
python3 continuous_paper_generator.py \
  --research-dir "$RESEARCH_OUTPUT_DIR" \
  --topic examples/example_topic.md \
  --paper-types icbinb
```

### C) Long-running daemon (recommended for continuous iteration)

```bash
python3 continuous_research_daemon.py \
  --source-config configs/sources/stable_source_priority.example.json \
  --duration-hours 24 \
  --enable-rewrite-followup \
  --auto-source-quality-feedback \
  --auto-quality-strategy-feedback \
  --auto-quality-governor \
  --auto-evidence-strategy-feedback \
  --auto-export-submission-dossier \
  --auto-failure-guard \
  --serve-dashboard \
  -- --submission-mode --num-ideas 3
```

Common ops commands:

```bash
bash run_stable_daemon.sh status
bash run_stable_daemon.sh brief
bash run_stable_daemon.sh handoff
bash run_stable_daemon.sh report-trends
bash run_stable_daemon.sh source-plan
```

---

## Outputs & Observability

XScientist writes structured artifacts under the output root (directory names may evolve across versions):

- `projects/`: full per-project directories
- `experiments/`: experiment outputs and logs
- `ideas/`: idea artifacts
- `reports/`: trends/handoff reports (daemon)
- `knowledge_base/`: cross-project memory (e.g., self-evolution history/playbook)

Common index/board commands (see `research_manager.py --help` for more):

```bash
python3 research_manager.py rebuild-index
python3 research_manager.py submission-board --top 5 --require-gate
python3 research_manager.py rewrite-board --top 10
python3 research_manager.py repair-board --top 20 --priority-tier p0
python3 research_manager.py evolution-board --top 20
python3 research_manager.py process-board --status blocked --top 30
```

---

## Docs

- `docs/guides/PROJECT_USAGE.md`: `run_project.py` usage and flags
- `docs/CONFIG_REFERENCE.md`: detailed configuration and parameters
- `docs/SOURCE_ORCHESTRATION.md`: source queue orchestration and recommended run postures
- `docs/LOGIN_GUARDRAIL.md`: login guard and session management
- `docs/guides/OUTPUT_DIRECTORIES.md`: output directory policy (if it diverges from code, follow `ai_scientist/config/paths.py`)

Note: some docs are currently Chinese-first.

---

## Development

- Unit tests: `make test`
- Syntax/import/validation smoke: `make smoke`
- Stricter local doctor: `make doctor` (requires a valid login session)
- Formatting: `make format`

---

## Roadmap

> Issues and PRs welcome (see `CONTRIBUTING.md`).

- [ ] Integrate TODO closure signals into governor phase switching and regression detection
- [ ] Strengthen evidence metrics and binding to experiment outputs (figure/table/metrics)
- [ ] Add consistency/regression checks for a "submission-ready" dossier
- [ ] Wire self-evolution/playbook more directly into automatic rewrite / follow-up executors
- [ ] Provide more complete English docs and API-style docs for external contributors

---

## Contributing & Community

- Contributing guide: `CONTRIBUTING.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`

---

## License

Apache-2.0. See `LICENSE`.

---

## Acknowledgements

Thanks to the open-source projects that inspired parts of this work:

- [Sakana AI: AI Scientist](https://github.com/SakanaAI/AI-Scientist)
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [awesome-ai-research-writing](https://github.com/Leey21/awesome-ai-research-writing)
- [AIDE](https://github.com/WecoAI/aideml)
- [DeepReviewer-v2](https://github.com/ResearAI/DeepReviewer-v2)

---

## Citation

If you use XScientist in research, please cite it (and include a commit hash for reproducibility):

XScientist Board (paper, authored with this system):

```bibtex
@misc{xscientist_board,
  title  = {XScientist Board: Artifact-Routed Submission Hardening for Autonomous Research Systems},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {NeurIPS 2026 submission (under review)}
}
```

XScientist (software / repository):

```bibtex
@software{xscientist,
  title  = {XScientist},
  author = {smileformylove and contributors},
  year   = {2026},
  url    = {https://github.com/smileformylove/ai_scientist}
}
```
