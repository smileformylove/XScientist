# Getting started

## 1. Prove the installation works at zero cost

```bash
xscientist demo ./first-study --autopilot
xscientist status ./first-study
xscientist benchmark first-run --max-seconds 30
```

Expected outcome: an offline DAG, deterministic runtime receipts, zero model
cost, and a next action that asks the user to resolve a contested claim.

## 2. Prepare a model-backed workspace

Install the 0.1.3 release candidate from `main` with exactly one provider
extra. For a repeatable experiment, replace `main` with a commit hash:

```bash
python -m pip install \
  "xscientist[research,openai] @ git+https://github.com/smileformylove/XScientist.git@main"
xscientist start ./my-study
```

The latest published PyPI release is `0.1.2`; it does not yet include every
journey documented on this page.

In a terminal, missing inputs are prompted progressively: question, ready
provider, model, evidence mode, and optional cost limit. Automation should pass
the same choices explicitly with `--non-interactive`.

Before spending money:

```bash
xscientist provider check --workspace ./my-study --max-cost-usd 10
xscientist executor prepare --workspace ./my-study
xscientist doctor --workspace ./my-study --deep
```

Credential presence and live API validation are different states. The provider
check does not make a paid request.

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

## 4. Keep the installation healthy

```bash
xscientist upgrade check --workspace ./my-study
xscientist upgrade check --workspace ./my-study --online
source <(xscientist completion zsh)
```

Upgrade checks are read-only and offline unless `--online` is present. Shell
completion is printed to stdout and never edits a shell profile.
