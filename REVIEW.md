# Review Guidance

Guidance for contributors and review agents, inspired by Anthropic's
[How the Claude Code team does code review](https://claude.dev/blog/solving-code-review-at-anthropic/)
(2026-10-15). This file does not install an AI reviewer or configure GitHub
branch protection. Anthropic's reported improvements are not XScientist results.

## What Matters Here

Prioritize incorrect scientific claims, false passes, lost evidence/history,
unsafe execution, credential exposure, and unintended cost or publication.
Style is actionable only when it obscures correctness or violates a checked
project convention. Report pre-existing problems separately from regressions.

Every finding should include severity, file/line, the failing behavior, and a
reproduction or concrete evidence. Start the review with an actionable finding
count; say explicitly when none were found. Never invent a test result.
Resolve each finding with a fix or an explicit, evidence-backed reason.

## Verification Before Approval

Run focused regression tests before requesting review. Use `make engineering`
for repository checks, and broaden tests for changes to shared contracts. Record
the exact commands and results, checks not run, and what the checks cannot prove.
Review only the intended change; do not include unrelated workspace edits.

Review effort follows risk, not change size or author identity:

| Risk | Examples | Review expectation |
| --- | --- | --- |
| Low | Documentation or isolated presentation | Check accuracy, links, and targeted behavior |
| Medium | Shared review/repair or routing behavior | Regression tests and downstream contract checks |
| High | Credentials, execution isolation, scientific gates, schemas, spending, publication, CI/releases | Human scrutiny plus negative and compatibility tests |

An agent's judgment does not substitute for required human approval. Record the
reviewed revision and rerun affected checks when it changes. Do not authorize
spending, execution, publication, or a weaker gate merely to clear a review.

## Research Review Closure

`ai_scientist/utils/review_jobs.py` maintains one issue per normalized text,
with reporting observations in `review_scopes` keyed by role, lane, and suite.
A completed review replaces only its own observations; a shared finding stays
active until every reporting scope clears it. Findings without a known legacy
owner are retained rather than assigned to an unrelated reviewer.

Closure requires a structured review with a `Weaknesses` list, either directly
(the native LLM reviewer output) or inside the `review` wrapper. Other
issue fields, when present, must also be lists of non-empty strings. An empty
list is an explicit clean result for that field; an omitted issue field does not
clear its findings. A missing/malformed payload or `missing_pdf` job cannot clear
issues. Reopened issues leave the resolved set while retaining
their first-seen metadata and appearance count. Lane summaries and hostile
findings exclude observations from other lanes.

With `AI_SCIENTIST_REPAIR_VERIFY=1`, reviewer rechecks in
`ai_scientist/utils/repair_verifier.py` require an explicit resolved record and
`resolution_checks` referencing matching, completed, valid review jobs. Hostile
rechecks additionally require hostile-review coverage. Missing/unknown issues
and legacy absence-only logs do not pass: rerun the responsible reviews to
record scoped closure. This is intentionally stricter than the old verifier.

## Version-Bound Review Inputs

`begin_review_job(..., pdf_path=...)` captures a portable `review_subject` before
the runtime reads the document. It binds the non-empty PDF bytes and any LaTeX
source declared in `manuscript_state`. Finalization checks those inputs again;
missing inputs and changes during review cannot produce a passing runtime result.
Findings from an unbound attempt are retained, but cannot close existing issues.
Passing a PDF only at finalization cannot retroactively bind an earlier review.

For a project that is itself a Git repository, the subject also records `HEAD`
and a content fingerprint of tracked working-tree files, including staged new
files, unstaged edits, deletion, and executable flags. Generated pipeline
contracts, `reviews/`, and `.ara/` are excluded to avoid self-invalidation.
Untracked source is not covered: add it to Git before requesting code review.
A non-Git run directory does not inherit an ancestor application's commit and
has manuscript-only binding. External paths, unsupported files, unavailable Git,
or hashing limits (256 MiB, 10,000 source files) fail closed.

Every closure check must reference the same subject version. The verifier hashes
the current inputs again, so an old pass cannot clear a changed document or code
revision. The next valid review reopens stale closures, restores all responsible
review scopes, and preserves the original jobs for audit. Partial passes from
different versions cannot be combined. Legacy closures without subject bindings
need fresh reviews, not synthesized hashes for historical jobs.

Persisted multi-role suites and primary/independent critic confirmation also
require one shared, current subject. The runtime rechecks the selected PDF path
so keeping the old file alongside a newly selected output cannot preserve a pass.

The target is the recorded PDF path, not whichever unrelated PDF happens to be
newest. A caller validating a different selected output must pass its `pdf_path`
in the repair task; a mismatched target fails. These local records are structural
checks, not signed credentials or proof of independent scientific reproduction,
PDF validity, or that experimental code actually produced its claimed results.
Figure bindings and claim-graph support alone do not establish scientific truth.

## Convergence And Next Work

Re-review should focus on fixes and regressions in the new diff, retain unresolved
findings, and identify bug classes instead of emitting the same comment repeatedly.
After three unsuccessful rounds, reassess the approach and evidence rather than
silently extending the loop. These are review instructions, not a new runtime cap;
existing workflow round and retry budgets still govern execution.

Next implementation priorities are automated risk-based routing and incremental
re-review with explicit coverage
(an unexamined issue must never be treated as cleared). Require a migration path
and regression cases before changing closure semantics again.

Turn incidents and false approvals into regression tests. Track false closures,
reopened issues, review rounds, cost, and human intervention alongside escaped
defects; fewer review rounds are not an improvement if failures increase.
