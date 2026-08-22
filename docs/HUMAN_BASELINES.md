# External human baselines (source-audited)

**Audit date: 2026-08-22.** This is a curated inventory of public primary
papers and official project pages that I could inspect. It is not a claim of
mathematical exhaustiveness over the entire internet. A row is included only
when its source and scope can be checked; an unreported result is written as
`not reported`, never as zero and never as an inferred human score.

This document exists because “human baseline” is used for several different
things in the literature. Mixing them produces a misleading leaderboard.

## Evidence classes

| Class | What must be true | What it does **not** mean |
| --- | --- | --- |
| `measured_human` | People actually attempted the stated tasks and the paper reports a score, time, or completion result. | It is not automatically comparable to another benchmark; task slice, budget, tools, and metric still matter. |
| `measured_human_small_sample` | People attempted a declared sample, but participant details, repetition, or coverage are limited. | It must not be extrapolated to the full benchmark or called an expert population without evidence. |
| `human_artifact_reference` | Human-produced artifacts (for example, public submissions) were executed or scored. | This is not a controlled, same-budget participant experiment. |
| `human_agent_uplift` | A controlled study measures people alone and people using an agent. | It is a process/usability result, not an autonomous-agent capability score. |
| `human_reference_proxy` | A leaderboard, public submission, or previously published result is used as a human reference. | It is not a recruited, same-condition participant arm. |
| `human_expert_artifact_reference` | A known expert/paper solution artifact is retained as a calibration or upper-bound reference. | It is not a distribution of human attempts or a controlled participant baseline. |
| `human_SOTA_reference` | A prior literature SOTA number is normalized as “human SOTA”. | `1.0` is not a newly measured human score. |
| `expert_validation_only` | Experts validate tasks, rubrics, programs, or judge labels. | Expert validation is not expert task performance. |
| `human_judgment_calibration` | People rate model outputs to test evaluator agreement. | Annotator agreement is not a human completion baseline. |
| `human_ground_truth_only` | Human reports or labels define the reference outcome. | Ground truth is not an independent human run under benchmark conditions. |
| `annotator_validation_reference` | Annotator answers are used to validate questions or estimate a reference score. | It is not necessarily an independently recruited, same-condition human benchmark arm. |
| `not_reported` | The audited source does not report a task-performance human arm. | This is not evidence that humans would score zero or that no other study exists. |

## Directly measured human task performance

These are the only rows in this inventory that can honestly be called direct
human performance measurements. Even these rows are not interchangeable.

| Benchmark / primary source | Human arm and controls | Reported human result | Comparability boundary |
| --- | --- | --- | --- |
| [RE-Bench](https://arxiv.org/abs/2411.15114) ([tasks](https://github.com/METR/RE-Bench)) | 61 distinct ML experts, 71 attempts; seven open-ended ML research-engineering environments; 8-hour attempts; same starting solution, scoring function, and 1–6 H100 setup as agents. | 82% of expert attempts achieved a non-zero score; 24% matched or exceeded the strong reference solution. The paper reports best-of-k/time-horizon curves, not one universal human percentage. | Strong direct comparison for short ML research engineering tasks. It is not a full scientific-discovery or months-long research baseline. |
| [PaperBench](https://arxiv.org/abs/2504.01848) | Eight enrolled/completed ML PhDs; human arm covers four papers with three independent attempts per paper; single A10 (four attempts used A100); part-time work over a four-week window with active time tracked. | On the three-paper subset, human best@3 reached **41.4% after 48 hours**; this is the paper’s subset aggregate, not all 20 papers. | Direct expert baseline for paper reproduction. The 41.4% number must not be copied to an unrelated task set or treated as a general “human research score”. |
| [DiscoveryWorld](https://arxiv.org/abs/2406.06769) ([official environment](https://github.com/allenai/discoveryworld)) | 11 practicing natural scientists (MSc/PhD); 16 normal/challenge tasks; same seed 0; maximum one hour per task; 2-D GUI; no retry-based zero-shot comparison. | Across the 16 human tasks: procedural **0.79**, completion **0.66**, knowledge **0.55**, average 717 steps. | A real measured baseline, but for a simulated interactive world and a one-hour protocol. It is not directly comparable to XScientist’s research-code workflow. |
| [DSBench](https://arxiv.org/abs/2409.07703) (`measured_human_small_sample` for analysis; `human_artifact_reference` for modeling) | Data-analysis result is based on **10 randomly sampled** challenges; the paper also executes usable human-submitted Kaggle code from 22 modeling competitions. The source does not clearly report human participant count/qualifications for the 10-task labeling sample. | Data-analysis subset: task-level accuracy **64.06%**, competition-level accuracy **67.33%**, average time **1107.7 s**. The modeling table reports human RPG **65.02** from the 22-competition code subset. | Useful reported subset, not the full 466-task suite. Because participant metadata are incomplete, do not call it an “expert baseline”. The modeling number is a public-artifact reference with maintenance/selection caveats, not a controlled participant score. |
| [BAISBench v1](https://arxiv.org/abs/2505.08341v1) (`measured_human_small_sample`) | Five PhD-level bioinformaticians each answered one random fifth of the **198-question** BAIS-SD set; one human expert also ran the 31-dataset cell-type annotation task with CellTypist. | BAIS-SD human correctness **0.762**; human-with-CellTypist BAIS-CTA **0.437 ± 0.014** (full set; supported subset **0.408 ± 0.013**). | These numbers are explicitly **v1** (13 May 2025). The later v2 changes the task size and protocol, so v1 scores must not be copied to v2 or to XScientist. |
| [BAISBench v2](https://arxiv.org/abs/2505.08341) (`measured_human_small_sample`, score not tabulated) | Six graduate-level bioinformaticians: one completed BAIS-DPTA and five collectively covered the **193-question** BAIS-SD set. | A human arm is confirmed, but the audited v2 text exposes the aggregate human BAIS-SD result only as a plotted Figure 6 bar, not a tabulated number; this inventory records the exact score as **not reported** rather than reading a pixel height. | Version drift is material (15 datasets/193 questions vs v1’s 31/198). A figure-derived approximation is deliberately not promoted to a precise baseline. |
| [BrowseComp](https://arxiv.org/abs/2504.12516) ([official report](https://openai.com/index/browsecomp/)) (`measured_human_small_sample`) | Human trainers (participant count not reported) attempted 1,255 of 1,266 browsing questions, could not use AI assistants, and could give up after two hours; they were not allowed to solve questions they created. | **367/1,255 = 29.2%** were solved. Of those, 317/367 (**86.4%**) matched the reference answer. The strict all-attempt match rate, derived from the same counts, is **25.3%** (317/1,255); it is not the paper’s headline metric. | A useful retrieval/persistence baseline with a hard two-hour censoring rule, not a scientific-discovery score. Trainer population and task assignment are not a controlled researcher sample; do not call 29.2% “human accuracy”. |
| [BrowseComp-V³](https://arxiv.org/abs/2602.12876) (`measured_human_small_sample`, participant count not reported) | 300 multimodal browsing questions; PhD-level participants used a standard browser and public web, with up to 30 minutes per question and an option to stop while recording exploration. | Pass@1 success rate **68.03%**; process score **82.93%**. The paper’s category-level results are reported separately and should not be averaged with text-only BrowseComp. | A particularly relevant process reference because it scores intermediate sub-goals, but the participant count and sampling frame are not reported. It is visual web search, not end-to-end scientific experimentation. |
| [Mind2Web 2](https://arxiv.org/abs/2506.21506) (`measured_human_small_sample`) | A randomly selected **30-task** subset of the 130-task benchmark; seven participants, with each task attempted by three different people; live web browsing, and participants could stop after one hour or when no clear path remained. | Partial completion **0.79 ± 0.01**, success rate **0.54 ± 0.07**, Pass@3 **0.83**, mean time **18.40 ± 1.61 min**. | This is a direct human arm for Subset-30 only, not the full benchmark. It measures long-horizon, citation-backed web search rather than scientific experimentation; the paper also notes that human effort can be underestimated by early stopping or omitted steps. |
| [WebArena](https://arxiv.org/abs/2307.13854) (`measured_human_small_sample`) | Five computer-science graduate students each attempted one task sampled from 170 intent templates in a self-hosted web environment; the paper reports an average task time of 110 seconds but no common hard time cap. | Overall human success **78.24%** (information-seeking **74.68%**, other site-navigation/content/configuration **81.32%**). | A useful controlled web-interaction reference, but the sample is small, tasks are templated/self-hosted, and the protocol is not a scientific-research workflow. Do not compare this percentage directly with XScientist. |

The published numbers above are reported in the cited papers. They are not
measurements of XScientist and are not pooled into a single “human average”.

## Human-only vs human+agent process evidence

This is relevant to XScientist’s Git-like process, but it answers a different
question from “can an autonomous agent match a scientist?”

| Study | What was actually measured | Result | Correct interpretation |
| --- | --- | --- | --- |
| [CORE-Bench follow-up, *Life After Benchmark Saturation*](https://arxiv.org/abs/2606.26158) | Five experienced evaluators, 20 ML/social-science papers, 50 randomized reproduction runs; each person did five manual and five human-agent runs; both had a three-hour cap and standardized Docker/A40 environments. | Manual sessions lasted **2.11×** as long as human-agent sessions (SE 0.09, two-sided p=0.00176). | A measured workflow/uplift baseline. It does not provide an autonomous-agent-vs-human capability score for original CORE-Bench. |

## Human references that are **not** measured participant baselines

These sources are useful context, but their numbers must not be labelled as a
new human run.

| Source | Evidence class | What the source says | Why it is not a human score |
| --- | --- | --- | --- |
| [MLE-bench](https://arxiv.org/abs/2410.07095) | `human_reference_proxy` | 75 Kaggle competitions are compared with public Kaggle results. | Public contestants worked under heterogeneous conditions (often weeks/months), while the agent protocol is much shorter; there is no controlled recruited-human arm. |
| [MLRC-Bench](https://arxiv.org/abs/2504.09702) ([leaderboard](https://huggingface.co/spaces/launch/MLRC_Bench)) | `human_reference_proxy` | Seven ML conference-competition tasks normalize the historical top-human solution to 100 and the starter baseline to 0; the best tested agent closes 9.3% of that gap on average. | “Top human” is a public competition solution at the time of the original contest. Participant count, work time, and a same-condition human arm are not reported; 100 is a reference normalization, not a new human score. |
| [ResearchGym](https://arxiv.org/abs/2602.15112) ([code](https://github.com/Anikethh/ResearchGym)) | `human_expert_artifact_reference` | Five recent paper environments/39 sub-tasks retain the source authors’ solution as a “soft upper bound” or known human-expert reference; the paper evaluates agents against it. | The source does not recruit people to run the packaged environment or report a human score distribution, participant count, or human budget. The reference artifact is not a measured human baseline. |
| [AIRS-Bench](https://arxiv.org/abs/2602.06855) | `human_SOTA_reference` | 20 ML research tasks normalize a prior-literature “human SOTA” to 1.0; agents exceed it on 4 tasks and do not reach it on 16. | The reference is a result quoted from prior work, not people re-running these 20 tasks with a shared budget. Never call 1.0 “measured human performance”. |
| [AutoResearchEval](https://arxiv.org/abs/2608.14905) ([official repo](https://github.com/PrentisAI/AutoResearchEval)) | `expert_validation_only` / `not_reported` | 100 tasks and 800 trajectories; experts label a stratified 50-trajectory sample; artifact-aware judge agreement is reported (pattern κ=0.75, taxonomy κ=0.83). | The audited paper reports human annotation/calibration, not a human task-performance arm. Its 50-trajectory labels are not a human baseline. |
| [ScienceAgentBench](https://arxiv.org/abs/2410.05080) | `expert_validation_only` / `not_reported` | 102 tasks from 44 papers; nine subject-matter experts validate tasks and provide knowledge; evaluators score generated programs. | The nine experts did not constitute a same-task human performance arm. Any reported agent score must not be relabelled as a human comparison. |
| [DeepResearch Bench](https://arxiv.org/abs/2506.11763) | `human_judgment_calibration` / `not_reported` | 100 PhD-level tasks; 50 Chinese tasks and four agent outputs per task were rated by three domain annotators; 70+ annotators were recruited and 37 tasks remained after an ICC filter for filtered correlation. | These are evaluator-consistency/correlation studies. The source does not report people completing the research tasks as a performance baseline. |
| [AstaBench](https://arxiv.org/abs/2510.21652) ([project page](https://allenai.org/asta/bench)) | `expert_validation_only` / `not_reported` | A large scientific-agent suite uses expert review and rubric/quality checks for its tasks and evaluations. | The audited source does not report a unified human participant run or human solve score; “human review” is not a human baseline. |
| [ResearchBench](https://arxiv.org/abs/2503.21248) | `expert_validation_only` / `not_reported` | Five PhD experts inspect decomposition/benchmark quality for 62 papers and report validation accuracy. | Validation of task extraction is not people performing the benchmark’s retrieval, hypothesis, or ranking tasks under a shared budget. |
| [CORE-Bench (original)](https://arxiv.org/abs/2409.11363) | `expert_validation_only` / `not_reported` | 270 reproducibility tasks from 90 papers; human checks establish that capsules are locally reproducible/feasible. | Human feasibility verification is not a measured human-vs-agent score. |
| [ReplicatorBench](https://arxiv.org/abs/2602.11354) | `human_ground_truth_only` / `not_reported` | 39 replication instances and 3,128 checkpoints use human expert replication reports as the reference outcome. | The reports define ground truth; the paper does not report recruited humans independently running the benchmark under the agent protocol. |

### “Not reported” is a valid result

For the linked AutoResearchEval paper, the honest record is:

```text
human_task_performance_baseline: not_reported_in_audited_source
human_role: trajectory_annotation_and_judge_calibration
human_annotation_is_a_score: false
```

This is intentionally different from `0`, `unknown quality`, or “the agent
beats humans”. The same rule applies to ScienceAgentBench, DeepResearch Bench,
original CORE-Bench, and ReplicatorBench.

## What XScientist can claim today

The local `benchmark autoresearch` pilot has **zero model rollouts and zero
human runs**. It measures manifest conformance, typed evidence visibility,
closure gates, and a bounded Git-like process view. It therefore has no
human-vs-agent scientific score and must keep `official_comparable: false`.
The external numbers above must not be inserted into the XScientist report as
if they were measured on the local manifest.

The honest near-term comparison is process-level: whether a human or agent run
leaves inspectable checkpoints, branches, evidence links, failed attempts,
repair decisions, and verifier results. That comparison still needs a matched
human arm; the process view makes the arm auditable but does not create one.

## Minimum record for a future XScientist human arm

Before publishing a comparison, store one record per run with at least:

| Field | Required content |
| --- | --- |
| `source_url` / `accessed_at` | Primary paper or registered study and retrieval date. |
| `evidence_class` | One of the classes above; `not_reported` is allowed. |
| `task_manifest_sha256` / `task_slice` | Same manifest and explicitly named slice for humans and agents. |
| `starting_artifact` / `tools_data_network` | Exact starting state and allowed resources. |
| `participants_n` / `attempts_n` | Participant and run counts, including exclusions. |
| `wall_clock_budget` / `compute` / `cost` | Time, hardware, provider/model cost, and unattended time policy. |
| `metric` / `score` / `uncertainty` | Verifier-defined metric, raw score, and interval or per-run distribution. |
| `raw_artifact_url` / `manifest_hash` | Consent-compatible artifact bundle, immutable hash, and evaluator version. |
| `comparability_status` | `matched`, `partial`, or `not_comparable`, with a reason. |

The human process record should contain observable work products and decisions,
not private free-form thought. Randomize or pre-register task order and stopping
rules, report individual runs rather than only a best run, and preserve the
same evaluator for both arms.

These requirements follow the reporting guidance in [Recommendations and
Reporting Checklist for Rigorous & Transparent Human Baselines](https://arxiv.org/abs/2506.13776),
which specifically calls for the same test set (or an explicitly matched
subset), participant/sample details, uncertainty, and enough material for
independent review.

## Adjacent general-agent baselines (not XScientist task baselines)

For completeness, the following are public human or annotator reference
measurements from widely used general-agent or reasoning benchmarks. They are
kept separate because a short question-answering or visual puzzle score does
not measure long-horizon scientific exploration.

| Benchmark / source | Human result actually reported | Scope and warning |
| --- | --- | --- |
| [GPQA](https://arxiv.org/abs/2311.12022) (`measured_human_small_sample`) | On the 448-question main set, expert-validator accuracy is reported as 71.9% and non-expert-validator accuracy as 30.4%; the paper’s extended-set headline is 65% expert accuracy (74% after conservative post-hoc correction). | Graduate-level multiple-choice science questions, with roughly 30–37 minutes and web access for non-experts. This is a knowledge/oversight baseline, not an autonomous research trajectory. |
| [GAIA](https://arxiv.org/abs/2311.12983) (`annotator_validation_reference`) | The paper reports an estimated 92% human score (including level-specific values 94/92/87) on its validated questions. | General assistant questions, not scientific experiments. The paper derives this number from annotator answers on valid questions; it does not report an independently recruited, controlled long-horizon researcher study. |
| [H-ARC](https://arxiv.org/abs/2409.01374) (`measured_human`) | 1,729 crowd workers; three-shot empirical average 76.2% on the 400-task training set and 64.2% on the 400-task public evaluation set, with reported uncertainty ranges. | Visual abstraction/program-synthesis puzzles. The source releases action traces, but the task and metric are unrelated to XScientist research artifacts. |
| [SciFIBench](https://papers.neurips.cc/paper_files/paper/2024/file/217bb44ab14621754db8a392163e6b07-Paper-Datasets_and_Benchmarks_Track.pdf) ([project page](https://SciFIBench.github.io/)) (`measured_human_small_sample`) | Five undergraduate/postgraduate participants answered a randomly sampled 25-question-per-task CS subset: mean **86.4%** Figure→Caption and **78.4%** Caption→Figure. | Scientific-figure multiple choice, not end-to-end research. The paper flags small/non-representative sampling and possible learning across the 50-question survey; there was no shared time limit. |

These rows are evidence that human baselines can be measured carefully; they
are not candidate numbers for an XScientist leaderboard.

## What is stored here

This repository stores the classification, quoted aggregate, protocol caveat,
primary URL, and audit date. It does **not** silently download papers, private
participant data, benchmark answers, or raw human trajectories into the local
pilot. The external source remains the evidence authority. If an immutable
replication package is needed, save a permitted source snapshot and its hash
outside the pilot and record that artifact in the future fields above; a URL in
this document is not a claim that the raw source is archived locally.

## Audit trail and update policy

- Primary sources were checked for participant count, task slice, budget,
  metric, and whether the human work was performance or evaluation.
- Secondary summaries are not used to upgrade a row’s evidence class.
- A source can move from `not_reported` to `measured_human` only after a primary
  source reports an actual human task run with enough protocol detail.
- No cross-benchmark average, rank, or “human frontier” number is maintained.
- When a source is ambiguous, XScientist records the more conservative class and
  links the ambiguity rather than filling a missing field.
