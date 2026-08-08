# Executable Self-Evolution Runtime

The evolution gate defines *what evidence is sufficient*. The evolution
runtime produces that evidence from real files and shell-free processes, then
allows a production directory to change only after semantic validation and
signed authorization both pass.

## Safety model

- Candidate sources are captured as immutable SHA-256 CAS objects. Symbolic
  links, special files, path traversal, undeclared diffs, and protected
  component scopes are rejected.
- Benchmark and canary commands are JSON argv arrays and never run through a
  shell. Execution is disabled unless `--execute` is present.
- A canary is deployed below one explicit deployment root. Every canary run
  restores the exact baseline artifact in a `finally` path; rollback failure
  makes the whole run fail.
- Production deployment defaults to a plan. `--apply` additionally requires a
  semantic promotion, candidate/evaluator/canary signatures, independent human
  approval signatures, and a `human:` approval identity.
- Production rollback requires a separate `production_rollback` signature.
  Existing releases and failed swaps are retained under
  `.xscientist-deploy/`; the adapter does not silently delete them.

The local adapter is deliberately bounded to directory deployment. Container,
Kubernetes, hosted-agent, HSM, transparency-log, and remote benchmark-custody
integrations should implement the same receipt and authorization interfaces.

## 1. Build a real candidate

Create `candidate-build.json`:

```json
{
  "base_root": "./baseline",
  "candidate_root": "./candidate",
  "candidate_id": "search-policy-v2",
  "component_type": "search_policy",
  "base_version": "1.0.0",
  "candidate_version": "1.1.0",
  "proposed_by": "agent:evolution",
  "change_summary": "Allocate more budget to falsification branches.",
  "change_scope": ["search/falsification.json"],
  "applicability_domains": ["general"],
  "failure_taxonomy_refs": ["failure:premature-convergence"],
  "ablation_dimensions": ["falsification-budget"]
}
```

Both source roots contain the logical component directory (`search/` in this
example). Build and persist both file trees:

```bash
xscientist evolution candidate \
  --spec candidate-build.json \
  --constitution science_constitution.json \
  --store .xscientist/evolution-cas \
  --out candidate.json
```

The command refuses a candidate when the actual added, modified, or removed
paths are not exactly covered by `change_scope`.

## 2. Run paired shadow benchmarks

A benchmark suite identifies the producer and evaluator stacks and contains
opaque task hashes, frozen timestamps, evaluation layers, and one command per
task. The command receives these variables:

- `XSCIENTIST_ARTIFACT_ROOT`
- `XSCIENTIST_TASK_ID`
- `XSCIENTIST_VARIANT` (`baseline` or `candidate`)
- `XSCIENTIST_OUTPUT`

It writes a JSON object containing `metrics`, `safety_pass`, `integrity_pass`,
and `reproducibility_pass`. The same command is executed against both immutable
artifacts:

```bash
xscientist evolution benchmark \
  --suite sealed-suite.json \
  --candidate candidate.json \
  --store .xscientist/evolution-cas \
  --execute --out benchmark-run.json
```

Stdout and stderr are represented by hashes in the run receipt. Secret-like
environment variable names are rejected instead of copied into the evaluator.
The resulting `samples` are direct inputs to `xscientist evolution-gate`.

## 3. Run a bounded canary and rollback exercise

Each canary project declares baseline `error_rate` and `quality` plus a
shell-free command. Its output adds `observations`, `incidents`, and the
long-tail, common-mode, and out-of-distribution verdicts.

```bash
xscientist evolution canary \
  --suite canary-suite.json \
  --candidate candidate.json \
  --store .xscientist/evolution-cas \
  --deployment-root ./controlled-canary \
  --executed-by service:canary-runner \
  --execute --out canary-run.json
```

The output separates the gate-compatible `canary_report` from the richer
deployment, project-run, and rollback receipts.

## 4. Sign evidence and approvals

New trust-boundary artifacts use `xscientist.canonical-json.v1`. The repository
ships the non-Python consumer
`ai_scientist/protocol/conformance/verify-canonical-json.mjs` and shared
test vectors. HMAC-SHA256 is available in the core runtime for controlled local
services; Ed25519 is available with `pip install "xscientist[trust]"`.

Signing secrets are read from an environment variable or private-key file and
are never written into an attestation:

```bash
export EVOLUTION_SIGNING_KEY="..."
xscientist evolution attest sign \
  --payload benchmark-gate.json \
  --purpose independent_benchmark \
  --identity service:independent-evaluator \
  --key-id evaluator-2026-08 \
  --key-env EVOLUTION_SIGNING_KEY \
  --out benchmark.attestation.json
```

A trust store maps a key ID to its identity and algorithm. HMAC entries name a
`key_env`; Ed25519 entries name a `public_key_file`. Production authorization
requires exactly one candidate, independent benchmark, and canary attestation,
plus the required independent `production_approval` attestations.

## 5. Plan, deploy, and roll back production

```bash
# No file mutation: validates promotion and all signatures, then prints a plan.
xscientist evolution deploy \
  --candidate candidate.json --promotion promotion.json \
  --constitution science_constitution.json \
  --authorization authorization-bundle.json --trust-store trust.json \
  --store .xscientist/evolution-cas \
  --deployment-root ./managed-production --target search \
  --executed-by service:release-adapter --approval human:release-owner

# Explicit mutation after the same checks. The optional Research VCS flags
# atomically record the verified production receipt against the promoted object.
xscientist evolution deploy ... --apply \
  --research-repo ./research --promoted-object rso-promoted-id \
  --out deployment-receipt.json
```

For production rollback, sign this payload with purpose
`production_rollback`: `candidate_hash`, `baseline_artifact_hash`, `target`,
and `trigger`. Then run:

```bash
xscientist evolution rollback \
  --candidate candidate.json --store .xscientist/evolution-cas \
  --deployment-root ./managed-production --target search \
  --executed-by service:release-adapter --approval human:release-owner \
  --trigger quality_regression --production \
  --authorization rollback.attestation.json --trust-store trust.json \
  --research-repo ./research --candidate-object rso-candidate-id \
  --promoted-object rso-promoted-id \
  --apply --out rollback-receipt.json
```

`ResearchEvolution.promote()` remains a semantic Research VCS operation.
Actual file mutation is represented only by a deployment adapter receipt; this
keeps scientific approval history separate from infrastructure authority. A
recorded production deployment is accepted only when its target, timestamp,
content hashes, backup reference, signer binding, and canonical receipt hash
all validate. `--no-commit` stages this evidence instead of checkpointing it.
