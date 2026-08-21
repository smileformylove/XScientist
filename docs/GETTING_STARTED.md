# Getting started

## 1. Start with your own idea at zero cost

No model, API key, Docker installation, or provider configuration is needed:

```bash
xscientist explore ./my-study
```

Answer only what you know. The command records the idea first, then guides you
through an expected observation, a disconfirming result, and a fair first test.
It does not invent missing answers, evidence, or conclusions.

To inspect a complete provider-free example as well:

```bash
xscientist demo ./first-study --autopilot
xscientist status ./first-study
xscientist benchmark first-run --max-seconds 30
```

Expected outcome: an offline DAG, deterministic runtime receipts, zero model
cost, and a next action that asks the user to resolve a contested claim.

## 2. Optionally add a model

Install the published 0.1.3 release with exactly one provider extra:

```bash
python -m pip install \
  "xscientist[research,openai]==0.1.3"
xscientist start ./my-study
```

In a terminal, the question saved by `explore` is reused. Remaining inputs are
prompted progressively: ready provider/model, evidence mode, and optional cost
limit. A local Ollama model needs no API key. Automation should pass the same
choices explicitly with `--non-interactive`.

Before spending money:

```bash
xscientist provider check --workspace ./my-study --max-cost-usd 10
xscientist executor prepare --workspace ./my-study
xscientist doctor --workspace ./my-study --deep
```

Credential presence and live API validation are different states. The provider
check does not make a paid request. When you explicitly approve one minimal
remote verification, use `xscientist provider check --workspace ./my-study
--live --json`; it reports model identity without storing response content.

## 3. Detach, inspect, and resume

```bash
xscientist start ./my-study \
  --question "Which mechanism best explains the held-out failure?" \
  --allow-synthetic-data --max-cost-usd 10 --detach

xscientist runs list --workspace ./my-study
xscientist runs watch RUN_ID --workspace ./my-study
xscientist runs logs RUN_ID --workspace ./my-study --tail 100
xscientist runs cancel RUN_ID --workspace ./my-study
xscientist runs resume RUN_ID --workspace ./my-study
```

Detached run metadata is local and private. Public run views omit the research
question and exact resume arguments.

## 4. Audit, save, and recover

The everyday history surface is intentionally small:

```bash
xscientist status ./my-study
xscientist history list ./my-study
xscientist history show ./my-study --commit HEAD
xscientist history diff ./my-study --from HEAD^ --to HEAD
xscientist audit ./my-study --level trace
xscientist audit ./my-study --level replay
xscientist audit ./my-study --level verify

xscientist history save ./my-study -m "record corrected measurement rule"
xscientist history rollback ./my-study --commit HEAD
```

`status` is the default review page: it shows the current checkpoint, pending
research changes, and the trace/replay/verify check ladder. `show` inspects one
checkpoint and `diff` explains the scientific change between two checkpoints.

The rollback command above is a read-only preview. It reports the exact target,
scientific impact, blockers, and an apply command. `--apply` appends a reversal
checkpoint instead of deleting history. It refuses tracked, staged, selected,
or policy-eligible unsaved research changes and never permits the repository's
first checkpoint to be reversed. Generated DAG views are preserved; if a view
represents an older checkpoint, `status` marks it stale and prints the refresh
command.

Use `xscientist research --help` only when you need branching, deep semantic
diffs, reproduction execution, bundles, or other protocol-level controls.

## 5. Keep the installation healthy

```bash
xscientist upgrade check --workspace ./my-study
xscientist upgrade check --workspace ./my-study --online
source <(xscientist completion zsh)
```

Upgrade checks are read-only and offline unless `--online` is present. Shell
completion is printed to stdout and never edits a shell profile.
