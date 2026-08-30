# Third-party notices and code lineage

This file records code lineage and attribution; it is not legal advice and
does not change any applicable license. XScientist-authored contributions are
released under the Apache-2.0 text in `LICENSE`. Modified or retained
third-party components remain subject to their original notices and licenses.
The distribution's SPDX expression is `Apache-2.0 AND MIT` because it contains
both Apache-2.0 lineage and modified AIDE-lineage code under MIT.

## AI-Scientist-v2

Parts of `ai_scientist/` are modified derivative code or retained assets from
the AI-Scientist-v2 lineage. A reproducible repository audit found the closest
fixed public comparison point to be:

- Project: [SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)
- Audited Apache-2.0 snapshot: `defddb8174905aac3bf4f7de7650e4cbf2ac353c`
- Snapshot license: [Apache License 2.0](https://github.com/SakanaAI/AI-Scientist-v2/blob/defddb8174905aac3bf4f7de7650e4cbf2ac353c/LICENSE)
- License SHA-256: `0dc1811264bb76e119d3f58a8e775d79abeb1708c9486d3861354c1688062350`
- Paper: [The AI Scientist-v2](https://arxiv.org/abs/2504.08066)

The complete Apache-2.0 text is distributed as `LICENSE`. The upstream
project changed its license after the audited snapshot in commit
[`c204ee8e7c68f7258284d82b666f6c06a76fef59`](https://github.com/SakanaAI/AI-Scientist-v2/commit/c204ee8e7c68f7258284d82b666f6c06a76fef59);
no future upstream code may be imported without a new provenance and license
review.

The public XScientist history does not establish the actual acquisition date
or exact acquired ref. Maintainers must complete those factual fields in
`provenance/upstream_sources.json`; this audit does not infer them.

Affected path families include:

```text
ai_scientist/ideas/i_cant_believe_its_not_better*
ai_scientist/llm.py
ai_scientist/vlm.py
ai_scientist/perform_icbinb_writeup.py
ai_scientist/perform_ideation_temp_free.py
ai_scientist/perform_llm_review.py
ai_scientist/perform_plotting.py
ai_scientist/perform_vlm_review.py
ai_scientist/perform_writeup.py
ai_scientist/tools/**
ai_scientist/treesearch/**
ai_scientist/utils/token_tracker.py
```

These components have been modified and extended for XScientist. XScientist
does not claim AI-Scientist-v2 benchmark parity or inherit its reported scores.

The previous full-paper review examples and bundled TeX packages/conference
style files from this lineage have been removed. The wheel does not redistribute
those papers, `natbib`, `fancyhdr`, `algorithm`, `algorithmic`, or historical
ICLR/ICML `.sty`/`.bst` files.

## AIDE

AI-Scientist-v2 states that its tree-search system was built on AIDE. The
corresponding lineage remains visible in parts of `ai_scientist/treesearch/`.

- Project: [WecoAI/aideml](https://github.com/WecoAI/aideml)
- Audited ref: `a4d58d94ad2035b7b458b5677c26a55e66ea8ca0`
- License: MIT, Copyright (c) 2024 Weco AI Ltd
- Full license: `third_party/licenses/AIDE-MIT.txt` (included in source and wheel)

Affected foundations include the interpreter, journal, model backends,
metric/response helpers, serialization, data preview and tree visualization.
They are modified components; XScientist does not claim AIDE benchmark results.

## XScientist-authored review calibration and manuscript seeds

All files currently distributed under `ai_scientist/fewshot_examples/` are
short, fictional XScientist-authored calibration examples. They are expressly
labelled as synthetic and do not reproduce a real paper or conference review.

The two packaged `blank_*_latex/template.tex` files are XScientist-authored
generic manuscript seeds, not official venue templates. Venue-targeted
workflows fetch the pinned current template directly from the official venue,
verify its SHA-256 digest, and write a source receipt into the user's research
workspace. Official venue template files are not bundled in the XScientist
source distribution or wheel and remain subject to the venue's own terms.

## Design references without imported code

The following sources are design or evaluation references. No code from these
projects is vendored by the corresponding XScientist implementation, and no
benchmark score is inherited: FAR, Faraday/Replica, Belief Context Graph,
AutoResearchEval, MLS-Bench, GEPA and Recuris. See
[the research-lineage documentation](https://github.com/smileformylove/XScientist/blob/main/docs/RESEARCH_LINEAGE.md)
for exact relationships and claim boundaries.
