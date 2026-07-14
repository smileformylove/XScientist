# arXiv Metadata Draft

Title:

XScientist: A Git-Like Research Protocol for Long-Running Autonomous Scientific Discovery

Authors:

Jixiang Luo

Contact email:

jixiangluo85@gmail.com

Abstract:

Autonomous research systems are often evaluated as one-shot paper generators: given a topic, they produce a manuscript and a small set of experiment logs. This framing hides the operational problem that makes such systems difficult to trust: research is long-running, branching, failure-prone, and dependent on auditable handoffs between agents and humans. XScientist is a git-like research protocol and operating system for this setting. It orchestrates idea generation, experiment execution, manuscript drafting, self-review, repair, quality gating, daemon scheduling, and reproducibility artifacts as one continuously observable pipeline. The central design choice is to treat each run as a portable research artifact rather than only as a PDF. XScientist exports an Agent-Native Research Artifact (ARA), a protocol that records an exploration DAG, per-node code and outputs, claim-to-evidence anchors, content hashes, provenance, and re-execution hooks. This makes each generated paper inspectable as a science exploration tree: failed branches, repaired experiments, ablations, and manuscript claims remain connected to the nodes that produced them. The system also includes deterministic integrity forensics, sample gates, truth contracts, reviewer-oriented repair loops, and long-running daemon controls. This paper describes the current XScientist architecture, the ARA protocol surface, and the practical safeguards needed to move autonomous science from single-run demos toward reproducible, reviewable, and forkable research infrastructure.

Suggested primary category:

cs.AI

Possible cross-list categories:

cs.SE

Comments:

System report and protocol proposal; source code available at https://github.com/smileformylove/XScientist

Repository source package:

`paper/xscientist_arxiv/main.tex`
