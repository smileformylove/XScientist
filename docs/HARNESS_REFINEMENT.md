# Evidence-guided harness refinement

Implementation checklist:

- [x] Opt-in shadow command with separate invariant and domain proposals.
- [x] Receipt-bound activation evidence, repeated baseline controls, and abstention.
- [x] Conflict-aware composition, fresh probes, and scoped experience records.
- [x] Held-out evaluation, execution/cost bounds, and local integration checks.

Validation: 85 refinement/evolution tests pass, including real Python
subprocess probes. The standalone smoke example completes 16 executions and
selects a composed shadow candidate after its held-out test. This validates
the implementation contract, not scientific performance.

The design adapts the Propose-Probe-Compose pattern from
[Harness2](https://github.com/google-research/harness2/tree/1cfe254e8d25b15f37048c3730134686abe60335).
It uses XScientist's existing immutable candidates and paired benchmark runtime.
No upstream code, model weights, benchmark data, or reported scores are imported.

Proposals remain atomic changes to one mutable component. Invariant proposals
address execution procedure; domain proposals address a scientific method.
Each proposal must declare a hypothesis, falsifier, applicability cue, and an
activation identifier that the evaluator can observe. Composition is a new
shadow candidate, requires its own execution, and never substitutes for an
attributed ablation or the existing production evolution gate.

Public development probes may guide composition. Hidden sealed/prospective
tasks are evaluated only after the development winner has been frozen. A failed
holdout returns to baseline rather than choosing another candidate on that
holdout. Experience remains quarantined and advisory until independent
historical and holdout backtests satisfy the existing skill admission rules.

## Run

The model-free smoke example creates a new directory, builds two real recovery
components, executes them in Python subprocesses, and saves the inputs and full
report. Use a directory that does not already exist:

```bash
python -m scripts.evolution.harness_refinement_demo --output /tmp/xsci-refinement-demo
```

The example repairs comma-separated numeric input and an arithmetic mean. Its
development and held-out inputs differ. It demonstrates runtime behavior, not
scientific discovery quality or benchmark improvement. Both proposer and
evaluator are local demo code; their distinct declared identities are not
independent external attestation.

Inspect or re-run the generated inputs with the product CLI:

```bash
xscientist evolution refine \
  --spec /tmp/xsci-refinement-demo/spec.json \
  --constitution /tmp/xsci-refinement-demo/constitution.json \
  --store /tmp/xsci-refinement-demo/cas

xscientist evolution refine \
  --spec /tmp/xsci-refinement-demo/spec.json \
  --constitution /tmp/xsci-refinement-demo/constitution.json \
  --store /tmp/xsci-refinement-demo/cas \
  --execute --out /tmp/xsci-refinement-demo/rerun.json
```

Without `--execute`, validation and planning perform no subprocess execution,
composition, model calls, memory updates, or deployment. Exit 0 means the plan
or experiment completed, including a valid `keep_base` outcome. Exit 3 means a
runtime budget or scientific gate halted execution; exit 2 means invalid input
or an invocation error.

## Proposal and evidence contracts

The spec uses `schema_version: xscientist.harness-refinement-spec.v1` and has
`proposals`, `development_suite`, optional `holdout_suite`, and optional `policy`.
Each proposal supplies:

- `proposal_id`: a unique bounded identifier; `composition` is reserved.
- `dimension`: `invariant` or `domain`.
- `candidate`: the inner candidate from `evolution candidate`.
- `hypothesis`, `falsifier`, `cue`: the proposed effect, rejection condition,
  and situation in which the procedure applies.

Proposal generation is explicit: an agent or person first creates ordinary
atomic candidate sources and CAS candidates. This command optimizes those
supplied proposals; it does not start another model session. All proposals
share one baseline, version, and mutable component. Actual file differences
must match the declared scopes. Duplicate candidate trees are rejected.

The evaluator emits the ordinary benchmark metrics and three scientific
gates, plus optional `behavior_evidence`:

```json
{
  "event_id": "parse-call-1",
  "change_id": "inv",
  "kind": "activated",
  "detail": "The evaluator observed the candidate parser consume the input."
}
```

Put events in a JSON array under `behavior_evidence` in the evaluator output.
`kind` is `activated`, `effect`, or `counterexample`. Events are bounded and
hashed inside each run receipt. Every constituent change must be observed in
every candidate run. These are evaluator-reported observations, not model
self-scores or cryptographic proof of causal attribution. The adapter must
instrument actual execution; reading a component's declaration is insufficient.

The optimizer uses public development metrics, unlike Harness2's ground-truth
free contrastive editor. A candidate must satisfy all scientific gates, retain
all five default evolution metrics, avoid per-pair metric regressions beyond
the existing default allowances, and beat the best repeated baseline objective
on each task by the configured margin. These conservative smoke criteria do
not establish statistical significance.

The best eligible proposal in each dimension may be composed when their actual
file changes do not overlap and their applicability domains intersect.
Composition supports additions, modifications and deletions, keeps the higher
risk tier, and is independently probed. An unsupported composition is excluded;
eligible original proposals remain available. Baseline is always the fallback.

The development selection receives a content hash before held-out execution.
Hidden task IDs and content hashes must be disjoint from development tasks.
Without a holdout, a development winner remains advisory and `decision` stays
`keep_base`. Failing a holdout never triggers selection of a runner-up.

## Budgets and authority

Default policy is two paired repeats per task, a 0.01 objective margin, at most
128 subprocess executions, and a reported cost limit of 100. The limits are
configurable using `repeats`, `minimum_improvement`, `max_executions`, and
`max_cost`. The planner reserves a composition and a final holdout in its
execution ceiling. Every candidate draw has a matched baseline draw; the
repeat index is supplied as `XSCIENTIST_REFINEMENT_REPEAT`. The evaluator must
use this index for any stochastic seed policy. Artifact roots are freshly
materialized for every pair.

The execution ceiling and per-process timeout are host-enforced. Cost comes
from the evaluator's `cost_per_task` metric in the same unit throughout the
suite and is checked after each pair; it can overshoot by one pair and is not
a provider-enforced spending cap. `accounting` separately counts baseline and
candidate executions and reported cost, plus observed process seconds.
Proposal-generation cost is unobserved because proposals are supplied inputs;
this is not a complete comparison to best-of-N at equal total model spend.
Missing cost, invalid output, or scientific
gate failure halts further probes and retains negative receipts. This uses the
existing local subprocess runner, not an OS sandbox. Untrusted generated code
still needs the configured isolated executor inside the evaluator command.

The report includes all attempts, metrics, receipt/event references, selected
candidate, selection commitment, cost, and positive/negative scoped experience.
Experience stays `quarantined`; no shared memory is updated. A selected result
is an ordinary immutable shadow candidate. Production use still requires
attributed ablations, sealed/prospective independent evaluation, canary,
rollback proof, and authorization through the existing evolution gate.
