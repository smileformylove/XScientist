# Modified by XScientist contributors from the AI-Scientist-v2/AIDE lineage.
# See THIRD_PARTY_NOTICES.md for provenance and license details.
import copy
import hashlib
import json
import math
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from .journal import Node, Journal

from ai_scientist.llm import get_response_from_llm, extract_json_between_markers
from ai_scientist.treesearch.backend import get_ai_client
from ai_scientist.utils.evaluation_binding import evaluation_hash_binding
from ai_scientist.utils.llm_budget import is_llm_budget_exception
from ai_scientist.utils.privacy import redact_sensitive_text

REPORT_ADVISORY_SCHEMA = "xscientist.report-advisory.v2"
REPORT_SUMMARY_SCHEMA = "xscientist.evidence-bound-stage-summary.v2"
PLOT_MANIFEST_SCHEMA = "xscientist.plot-evidence-manifest.v1"

ARTIFACT_BINDING_SCOPE = "artifact_identity_only_not_scientific_verification"
_PLOT_CAPTION = (
    "This plot is included as a host-bound artifact. Its scientific "
    "interpretation remains advisory."
)

_NARRATIVE_CLAIM_CATALOG = {
    "experiment.qualified_method_replay": (
        "Experiment_description",
        "The gate-qualified method was replayed using confirmation runs.",
    ),
    "experiment.held_out_confirmation": (
        "Experiment_description",
        "The reported evidence comes from the held out confirmation workflow.",
    ),
    "significance.reproducibility_scope": (
        "Significance",
        "The replay evidence helps assess whether the qualified result is internally reproducible.",
    ),
    "significance.internal_consistency_only": (
        "Significance",
        "The evidence establishes artifact internal consistency rather than independent scientific validation.",
    ),
    "description.deterministic_evidence_gate": (
        "Description",
        "The host accepted quantitative results only from the deterministic evidence gate.",
    ),
    "description.agent_interpretation_advisory": (
        "Description",
        "Agent authored interpretation is advisory and remains subject to external scientific review.",
    ),
}

_ADVISORY_FIELDS = {
    "schema",
    "Experiment_description",
    "Significance",
    "Description",
    "List_of_included_plots",
    "Key_numerical_results",
}
_FINAL_SUMMARY_FIELDS = _ADVISORY_FIELDS | {
    "stage_name",
    "Narrative_claim_ids",
    "Plot_evidence_manifest",
}
_FINAL_PLOT_FIELDS = {"plot_claim_id", "caption"}
_PLOT_MANIFEST_FIELDS = {
    "schema",
    "path_base",
    "artifact_binding_scope",
    "entries",
    "manifest_hash",
}
_PLOT_MANIFEST_ENTRY_FIELDS = {
    "plot_claim_id",
    "path",
    "content_sha256",
    "qualified_node_id",
    "evaluation_result_hash",
    "multi_seed_receipt_hash",
    "evaluation_verification_scope",
    "artifact_binding_scope",
}
_NUMERICAL_RESULT_FIELDS = {
    "dataset_name",
    "metric_name",
    "confirmation_mean",
    "ci95_lower",
    "ci95_upper",
    "ci95_half_width",
    "n",
    "sample_stdev",
    "standard_error",
    "minimum",
    "maximum",
    "qualified_node_id",
    "evaluation_result_hash",
    "multi_seed_receipt_hash",
    "verification_scope",
}

report_summarizer_sys_msg = """You are an expert machine learning researcher.
You are given only gate-qualified experiment evidence and the confirmation-seed
nodes bound to its validated receipt. Your task is to summarize that evidence.

Important instructions:
- Do NOT hallucinate or fabricate information that is not present in the logs.
- Do NOT introduce errors when repeating information from the logs.
- You are an advisory narrative writer, not verification authority.
- Select only opaque qualitative claim identifiers supplied by the host. Never
  write narrative prose, plot paths, captions, hashes, or quantitative claims.
- "Key_numerical_results" MUST be an empty list. The host derives every
  quantitative result from a validated multi-seed receipt after your response.
- Treat artifact-internal deterministic verification as an internal consistency
  check, not independent ground-truth validation.
- Treat agent-authored analysis as interpretation, not verification authority.
"""

output_format_control = """Respond in the following format:

THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```

In <THOUGHT>, thoroughly reason as an expert researcher. First, reason about each node, and then reason carefully by combining all the information. It is okay to be very detailed.

In <JSON>, provide the review in JSON format with the following fields in exactly this order:
- "schema": the literal string "xscientist.report-advisory.v2"
- "Experiment_description": a non-empty list of allowed claim identifiers for that section
- "Significance": a non-empty list of allowed claim identifiers for that section
- "Description": a non-empty list of allowed claim identifiers for that section
- "List_of_included_plots": a list containing only allowed plot claim identifiers
- "Key_numerical_results": an empty list. Never place an item in this list.

Do not add fields or copy source prose. Identifiers are opaque: reproduce only
identifiers shown in the allowed catalogs. Ensure the JSON is valid, as it will
be automatically parsed."""

report_summarizer_prompt = (
    """You are given multiple experiment logs from different "nodes". Each node represents attempts and experiments exploring various scientific ideas.

One key point is that these nodes collectively illustrate a stage of testing
different methods or approaches. Identify only qualitative scientific insights.
The host, not you, will render every quantitative result and confidence interval.
Leave "Key_numerical_results" empty and return claim identifiers instead of prose.

Be concise and avoid repeating the same information from different nodes. You are encouraged to be thorough, but you do not need to include information from every node. Reason carefully about which results from which nodes are scientifically insightful.

The name of this stage of the experiment: {stage_name}

Here are the experiment logs of the nodes:

{node_infos}

Allowed qualitative claim catalog:
{claim_catalog}

Allowed plot claim identifiers:
{plot_claim_ids}
"""
    + output_format_control
)

stage_aggregate_prompt = """You are given:

1) The summary of all previous experiment stages:
{prev_summary}

2) The name of the current experiment stage:
{stage_name}

3) The summary of the current stage:
{current_summary}


Your task is to produce an **updated comprehensive summary** of all experiment stages, including the newly introduced results from the current stage.

**Key Requirements:**
1. **No Loss of Critical Information**
   - Preserve valuable insights from the summary of all previous experiment stages. Do not remove or alter crucial texts.
   - Absolutely no hallucinations: if something does not appear in the logs or summaries, do not invent it. If something appears in the previous summary, do not make any mistakes when repeating it.
2. **Merge New Stage Data**
   - Integrate relevant results from the current stage into the existing summary.
   - Identify any overlap or repetition between new and old content, and remove only that which is clearly redundant or no longer scientifically insightful.
   - Be very careful if you want to remove or shorten the old content. By default, you can keep most of it and append new text.
   - Highlight how new findings connect to or differ from previous findings.
3. **Numerical Results and Visuals**
   - Carefully maintain the most insightful plots, figures, and numerical results.
   - Do not delete crucial quantitative findings or meaningful visual references.
4. **Length and Format**
   - The final summary will likely be **very long**. That is acceptable.
   - Present the updated summary in a format consistent with the style of the previous summaries (e.g., same section headings or structure).

Respond in the following format:

THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```
Ensure the JSON is valid and properly formatted, as it will be automatically parsed.
"""


def _strict_sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _qualified_evidence_root(node: Node) -> Path:
    raw_root = getattr(node, "exp_results_dir", None)
    if not isinstance(raw_root, (str, os.PathLike)) or not str(raw_root).strip():
        raise ValueError("Qualified plot evidence root is unavailable")
    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        raise ValueError("Qualified plot evidence root must be absolute")
    try:
        root_metadata = os.lstat(root)
        resolved = root.resolve(strict=True)
    except OSError:
        raise ValueError("Qualified plot evidence root is unavailable") from None
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("Qualified plot evidence root is not a regular directory")
    return resolved


def _resolve_regular_plot(path: Any, *, evidence_root: Path) -> tuple[Path, str]:
    """Resolve a non-symlink regular file relative to the qualified evidence root."""

    if not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise ValueError("Qualified plot evidence path is invalid")
    rendered = str(path).strip()
    if "\\" in rendered:
        raise ValueError("Qualified plot evidence path is invalid")
    source = Path(rendered).expanduser()
    if source.is_absolute():
        if ".." in source.parts:
            raise ValueError("Qualified plot evidence escapes its evidence root")
        candidate = source
    else:
        portable = PurePosixPath(rendered)
        if (
            portable.is_absolute()
            or not portable.parts
            or any(part in {"", ".", ".."} for part in portable.parts)
        ):
            raise ValueError("Qualified plot evidence escapes its evidence root")
        # Relative plot references are deliberately rooted here, never at cwd.
        candidate = evidence_root.joinpath(*portable.parts)

    candidate = Path(os.path.abspath(candidate))
    try:
        lexical_relative = candidate.relative_to(evidence_root)
    except ValueError:
        raise ValueError("Qualified plot evidence escapes its evidence root") from None
    if not lexical_relative.parts:
        raise ValueError("Qualified plot evidence path is invalid")

    cursor = evidence_root
    try:
        for part in lexical_relative.parts:
            cursor = cursor / part
            metadata = os.lstat(cursor)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("Qualified plot evidence must not use symlinks")
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(evidence_root)
        metadata = os.lstat(resolved)
    except ValueError:
        raise
    except OSError:
        raise ValueError("Qualified plot evidence file is unavailable") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("Qualified plot evidence must be a regular file")
    if relative != lexical_relative:
        raise ValueError("Qualified plot evidence changed during path validation")
    return resolved, relative.as_posix()


def _hash_regular_plot(path: Path) -> str:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ValueError("Qualified plot evidence file is unavailable") from None
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Qualified plot evidence must be a regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    try:
        current = os.lstat(path)
    except OSError:
        raise ValueError("Qualified plot evidence changed during hashing") from None

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
        )

    if (
        identity(before) != identity(after)
        or identity(after) != identity(current)
        or not stat.S_ISREG(current.st_mode)
    ):
        raise ValueError("Qualified plot evidence changed during hashing")
    return "sha256:" + digest.hexdigest()


def _plot_evidence_manifest(
    node: Node,
    *,
    evaluation_result_hash: str,
    multi_seed_receipt_hash: str,
    evaluation_verification_scope: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build a host-side artifact manifest; this is not scientific verification."""

    raw_paths = getattr(node, "plot_paths", None) or []
    analyses = getattr(node, "plot_analyses", None) or []
    if not isinstance(raw_paths, list) or not isinstance(analyses, list):
        raise ValueError("Qualified plot evidence is malformed")
    evidence_root = _qualified_evidence_root(node)

    path_evidence: dict[Path, tuple[str, str]] = {}
    for raw_path in raw_paths:
        resolved, relative = _resolve_regular_plot(
            raw_path,
            evidence_root=evidence_root,
        )
        if resolved in path_evidence:
            raise ValueError("Qualified plot evidence contains duplicate paths")
        path_evidence[resolved] = (relative, _hash_regular_plot(resolved))

    entries: list[dict[str, str]] = []
    source_context: dict[str, str] = {}
    used_paths: set[Path] = set()
    for item in analyses:
        if not isinstance(item, dict) or set(item) != {"plot_path", "analysis"}:
            raise ValueError("Qualified plot analysis is malformed")
        raw_path = item.get("plot_path")
        description = item.get("analysis")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Qualified plot analysis is malformed")
        resolved, relative = _resolve_regular_plot(
            raw_path,
            evidence_root=evidence_root,
        )
        file_evidence = path_evidence.get(resolved)
        if file_evidence is None or file_evidence[0] != relative:
            raise ValueError("Qualified plot analysis is not bound to allowed evidence")
        if resolved in used_paths:
            raise ValueError("Qualified plot analysis contains a duplicate path")
        used_paths.add(resolved)

        unsigned_entry = {
            "path": relative,
            "content_sha256": file_evidence[1],
            "qualified_node_id": node.id,
            "evaluation_result_hash": evaluation_result_hash,
            "multi_seed_receipt_hash": multi_seed_receipt_hash,
            "evaluation_verification_scope": evaluation_verification_scope,
            "artifact_binding_scope": ARTIFACT_BINDING_SCOPE,
        }
        plot_claim_id = "plot-claim:" + _strict_sha256_json(
            unsigned_entry
        ).removeprefix("sha256:")
        entry = {"plot_claim_id": plot_claim_id, **unsigned_entry}
        if set(entry) != _PLOT_MANIFEST_ENTRY_FIELDS:
            raise ValueError("Qualified plot evidence manifest entry is invalid")
        entries.append(entry)
        source_context[plot_claim_id] = redact_sensitive_text(description)

    entries.sort(key=lambda entry: (entry["path"], entry["plot_claim_id"]))
    manifest_without_hash = {
        "schema": PLOT_MANIFEST_SCHEMA,
        "path_base": "qualified.exp_results_dir",
        "artifact_binding_scope": ARTIFACT_BINDING_SCOPE,
        "entries": entries,
    }
    manifest = {
        **manifest_without_hash,
        "manifest_hash": _strict_sha256_json(manifest_without_hash),
    }
    if set(manifest) != _PLOT_MANIFEST_FIELDS:
        raise ValueError("Qualified plot evidence manifest is invalid")
    return manifest, source_context


def _verified_journal_nodes(journal: Journal) -> list[Node]:
    nodes = getattr(journal, "nodes", None)
    if not isinstance(nodes, list):
        raise ValueError("Report journal does not expose immutable node evidence")
    return [node for node in nodes if node.has_verified_metric]


def _validated_stage_evidence(
    journal: Journal,
    *,
    expected_stage_name: str | None = None,
) -> dict[str, Any]:
    """Revalidate the qualified receipt immediately before report generation."""

    verified_nodes = _verified_journal_nodes(journal)
    qualified_nodes = [
        node
        for node in verified_nodes
        if not node.is_seed_node and isinstance(node.multi_seed_report, dict)
    ]
    if len(qualified_nodes) != 1:
        raise ValueError("Report journal must contain exactly one qualified node")
    qualified = qualified_nodes[0]

    # Reuse the gate's canonical receipt and seed-link verification, while
    # intentionally ignoring rejected attempts that are absent from the final
    # qualified-only journal view.
    from .agent_manager import _validate_multi_seed_journal_links

    validation_node = copy.copy(qualified)
    validation_node.multi_seed_attempts = []
    try:
        report = _validate_multi_seed_journal_links(
            journal,
            validation_node,
            validation_node.multi_seed_report,
        )
    except Exception as exc:
        raise ValueError(f"Final report multi-seed receipt is invalid: {exc}") from None
    if expected_stage_name is not None and (
        not isinstance(expected_stage_name, str)
        or not expected_stage_name.strip()
        or report.get("stage") != expected_stage_name
    ):
        raise ValueError("Final report stage name does not match its evidence receipt")

    allowed_node_ids = {qualified.id} | {row["node_id"] for row in report["seeds"]}
    all_nodes = getattr(journal, "nodes", [])
    if (
        len(all_nodes) != len(allowed_node_ids)
        or {node.id for node in all_nodes} != allowed_node_ids
        or {node.id for node in verified_nodes} != allowed_node_ids
    ):
        raise ValueError("Final report input contains non-qualified evidence")

    evaluation_binding = evaluation_hash_binding(qualified.evaluation_report)
    evaluation_report = qualified.evaluation_report
    if (
        evaluation_binding is None
        or not isinstance(evaluation_report, dict)
        or evaluation_report.get("verification_scope")
        != "artifact_internal_consistency"
    ):
        raise ValueError("Qualified evaluation result hash is invalid")

    numerical_results: list[dict[str, Any]] = []
    metric_name = qualified.evaluation_comparison_contract["selected_metric"]
    for dataset_name in sorted(report["datasets"]):
        stats = report["datasets"][dataset_name]
        mean = float(stats["mean"])
        half_width = float(stats["ci95_half_width"])
        row = {
            "dataset_name": dataset_name,
            "metric_name": metric_name,
            "confirmation_mean": mean,
            "ci95_lower": mean - half_width,
            "ci95_upper": mean + half_width,
            "ci95_half_width": half_width,
            "n": stats["n"],
            "sample_stdev": float(stats["sample_stdev"]),
            "standard_error": float(stats["standard_error"]),
            "minimum": float(stats["min"]),
            "maximum": float(stats["max"]),
            "qualified_node_id": qualified.id,
            "evaluation_result_hash": evaluation_binding["result_hash"],
            "multi_seed_receipt_hash": report["receipt_hash"],
            "verification_scope": evaluation_report["verification_scope"],
        }
        if (
            set(row) != _NUMERICAL_RESULT_FIELDS
            or isinstance(row["n"], bool)
            or not isinstance(row["n"], int)
            or row["n"] < 3
            or any(
                not math.isfinite(row[key])
                for key in (
                    "confirmation_mean",
                    "ci95_lower",
                    "ci95_upper",
                    "ci95_half_width",
                    "sample_stdev",
                    "standard_error",
                    "minimum",
                    "maximum",
                )
            )
        ):
            raise ValueError("Qualified numerical report evidence is invalid")
        numerical_results.append(row)

    if not numerical_results:
        raise ValueError("Qualified report has no dataset evidence")
    plot_manifest, plot_source_context = _plot_evidence_manifest(
        qualified,
        evaluation_result_hash=evaluation_binding["result_hash"],
        multi_seed_receipt_hash=report["receipt_hash"],
        evaluation_verification_scope=evaluation_report["verification_scope"],
    )
    return {
        "qualified": qualified,
        "report": report,
        "plot_manifest": plot_manifest,
        "plot_source_context": plot_source_context,
        "numerical_results": numerical_results,
    }


def get_nodes_infos(nodes):
    """Render only advisory, privacy-safe context; omit all numerical results."""

    entries = []
    for node in nodes:
        report = (
            node.evaluation_report if isinstance(node.evaluation_report, dict) else {}
        )
        entries.append(
            {
                "node_role": "confirmation_seed" if node.is_seed_node else "qualified",
                "plan": redact_sensitive_text(
                    str(node.overall_plan or node.plan or "Not available")
                ),
                "analysis": redact_sensitive_text(
                    str(getattr(node, "analysis", None) or "Not available")
                ),
                "verification_scope": report.get("verification_scope"),
                "ground_truth_authority": report.get("ground_truth_authority"),
            }
        )
    return json.dumps(entries, ensure_ascii=False, sort_keys=True)


def get_summarizer_prompt(journal, stage_name):
    evidence = _validated_stage_evidence(
        journal,
        expected_stage_name=stage_name,
    )
    verified_nodes = _verified_journal_nodes(journal)
    node_infos = get_nodes_infos(verified_nodes)
    plot_context = [
        {"plot_claim_id": claim_id, "source_interpretation": description}
        for claim_id, description in sorted(evidence["plot_source_context"].items())
    ]
    node_infos += "\nHost-bound plot context:\n" + json.dumps(
        plot_context,
        ensure_ascii=False,
        sort_keys=True,
    )
    claim_catalog = {
        claim_id: {"section": section, "host_template": template}
        for claim_id, (section, template) in _NARRATIVE_CLAIM_CATALOG.items()
    }
    plot_claim_ids = [
        entry["plot_claim_id"] for entry in evidence["plot_manifest"]["entries"]
    ]
    return report_summarizer_sys_msg, report_summarizer_prompt.format(
        node_infos=node_infos,
        stage_name=stage_name,
        claim_catalog=json.dumps(
            claim_catalog,
            ensure_ascii=False,
            sort_keys=True,
        ),
        plot_claim_ids=json.dumps(plot_claim_ids, ensure_ascii=False),
    )


def _validate_claim_ids(value: Any, *, section: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("Report model must select qualitative claim identifiers")
    if (
        any(not isinstance(claim_id, str) for claim_id in value)
        or len(value) != len(set(value))
        or any(
            claim_id not in _NARRATIVE_CLAIM_CATALOG
            or _NARRATIVE_CLAIM_CATALOG[claim_id][0] != section
            for claim_id in value
        )
    ):
        raise ValueError(
            "Report model selected an invalid qualitative claim identifier"
        )
    return list(value)


def _validate_advisory_summary(
    summary: Any,
    *,
    plot_manifest: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(summary, dict) or set(summary) != _ADVISORY_FIELDS:
        raise ValueError("Report model output does not match the advisory schema")
    if summary.get("schema") != REPORT_ADVISORY_SCHEMA:
        raise ValueError("Report model output is schema-less or has a wrong schema")
    for field in ("Experiment_description", "Significance", "Description"):
        _validate_claim_ids(summary.get(field), section=field)
    if summary.get("Key_numerical_results") != []:
        raise ValueError("Report model must not provide numerical results")
    plots = summary.get("List_of_included_plots")
    allowed_plot_ids = {
        entry["plot_claim_id"] for entry in plot_manifest.get("entries", [])
    }
    if (
        not isinstance(plots, list)
        or any(not isinstance(claim_id, str) for claim_id in plots)
        or len(plots) != len(set(plots))
        or any(claim_id not in allowed_plot_ids for claim_id in plots)
    ):
        raise ValueError("Report plot claim is not bound to allowed evidence")
    return summary


def _render_claim_ids(claim_ids: list[str], *, section: str) -> str:
    validated = _validate_claim_ids(claim_ids, section=section)
    return " ".join(_NARRATIVE_CLAIM_CATALOG[claim_id][1] for claim_id in validated)


def _validate_completed_summary(
    summary: Any,
    *,
    stage_name: str,
    journal: Journal,
) -> dict[str, Any]:
    evidence = _validated_stage_evidence(
        journal,
        expected_stage_name=stage_name,
    )
    if not isinstance(summary, dict) or set(summary) != _FINAL_SUMMARY_FIELDS:
        raise ValueError("Completed report does not match the evidence-bound schema")
    if (
        summary.get("schema") != REPORT_SUMMARY_SCHEMA
        or summary.get("stage_name") != stage_name
    ):
        raise ValueError("Completed report has an invalid schema or stage binding")
    narrative_claim_ids = summary.get("Narrative_claim_ids")
    if not isinstance(narrative_claim_ids, dict) or set(narrative_claim_ids) != {
        "Experiment_description",
        "Significance",
        "Description",
    }:
        raise ValueError("Completed report narrative claim binding is invalid")
    for field in ("Experiment_description", "Significance", "Description"):
        claim_ids = narrative_claim_ids[field]
        expected_text = _render_claim_ids(claim_ids, section=field)
        if summary.get(field) != expected_text:
            raise ValueError("Completed report narrative is not host-template-bound")

    if summary.get("Plot_evidence_manifest") != evidence["plot_manifest"]:
        raise ValueError("Completed report plot manifest is not artifact-bound")
    allowed_plot_ids = {
        entry["plot_claim_id"] for entry in evidence["plot_manifest"].get("entries", [])
    }
    included_plots = summary.get("List_of_included_plots")
    if not isinstance(included_plots, list):
        raise ValueError("Completed report plot selection is invalid")
    selected_plot_ids: list[str] = []
    for plot in included_plots:
        if (
            not isinstance(plot, dict)
            or set(plot) != _FINAL_PLOT_FIELDS
            or plot.get("caption") != _PLOT_CAPTION
            or not isinstance(plot.get("plot_claim_id"), str)
            or plot["plot_claim_id"] not in allowed_plot_ids
            or plot["plot_claim_id"] in selected_plot_ids
        ):
            raise ValueError("Completed report plot selection is not manifest-bound")
        selected_plot_ids.append(plot["plot_claim_id"])
    if summary["Key_numerical_results"] != evidence["numerical_results"]:
        raise ValueError("Completed report numerical results are not evidence-bound")
    try:
        json.dumps(summary, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("Completed report is not strict JSON") from None
    return summary


def get_stage_summary(journal, stage_name, model, client):
    evidence = _validated_stage_evidence(
        journal,
        expected_stage_name=stage_name,
    )
    sys_msg, prompt = get_summarizer_prompt(journal, stage_name)
    response = get_response_from_llm(prompt, client, model, sys_msg)
    if (
        not isinstance(response, (list, tuple))
        or not response
        or not isinstance(response[0], str)
    ):
        raise ValueError("Report model returned no parseable advisory response")
    summary_json = extract_json_between_markers(response[0])
    advisory = _validate_advisory_summary(
        summary_json,
        plot_manifest=evidence["plot_manifest"],
    )
    narrative_claim_ids = {
        field: list(advisory[field])
        for field in ("Experiment_description", "Significance", "Description")
    }
    completed = {
        "schema": REPORT_SUMMARY_SCHEMA,
        "stage_name": stage_name,
        **{
            field: _render_claim_ids(narrative_claim_ids[field], section=field)
            for field in ("Experiment_description", "Significance", "Description")
        },
        "List_of_included_plots": [
            {"plot_claim_id": claim_id, "caption": _PLOT_CAPTION}
            for claim_id in advisory["List_of_included_plots"]
        ],
        "Key_numerical_results": evidence["numerical_results"],
        "Narrative_claim_ids": narrative_claim_ids,
        "Plot_evidence_manifest": evidence["plot_manifest"],
    }
    return _validate_completed_summary(
        completed,
        stage_name=stage_name,
        journal=journal,
    )


def get_node_log(node):
    node_dict = node.to_dict()
    # Only include keys that are relevant for logging/analysis
    keys_to_include = [
        "overall_plan",
        "analysis",
        "metric",
        "code",
        "plot_code",
        "plot_plan",
        "plot_analyses",
        "plot_paths",
        "vlm_feedback_summary",
        "exp_results_dir",
        "ablation_name",
    ]
    ret = {
        key: node_dict[key]
        for key in keys_to_include
        if key in node_dict and node_dict[key] is not None
    }
    if "exp_results_dir" in ret:
        original_dir_path = ret["exp_results_dir"]
        # Remove leading path segments before "experiment_results"
        idx = original_dir_path.find("experiment_results")
        short_dir_path = original_dir_path
        if idx != -1:
            short_dir_path = original_dir_path[idx:]

        ret["exp_results_dir"] = short_dir_path

        if os.path.isdir(original_dir_path):
            npy_files = [f for f in os.listdir(original_dir_path) if f.endswith(".npy")]
            # Prepend the shortened path to each .npy filename
            ret["exp_results_npy_files"] = [
                os.path.join(short_dir_path, f) for f in npy_files
            ]
        else:
            ret["exp_results_npy_files"] = []
    evaluation_report = (
        node.evaluation_report if isinstance(node.evaluation_report, dict) else {}
    )
    multi_seed_report = (
        node.multi_seed_report if isinstance(node.multi_seed_report, dict) else {}
    )
    ret["verification"] = {
        "metric_provenance": node.metric_provenance,
        "verification_scope": evaluation_report.get("verification_scope"),
        "ground_truth_authority": evaluation_report.get("ground_truth_authority"),
        "evaluation_result_hash": evaluation_report.get("result_hash"),
        "multi_seed_receipt_hash": multi_seed_report.get("receipt_hash"),
        "multi_seed_stage": multi_seed_report.get("stage"),
        "multi_seed_count": len(multi_seed_report.get("seeds", [])),
    }
    return ret


def update_summary(
    prev_summary, cur_stage_name, cur_journal, cur_summary, model, client, max_retry=5
):
    prompt = stage_aggregate_prompt.format(
        prev_summary=prev_summary,
        stage_name=cur_stage_name,
        current_summary=cur_summary,
    )
    try:
        response = get_response_from_llm(
            prompt, client, model, "You are an expert machine learning researcher."
        )
        summary_json = extract_json_between_markers(response[0])
        assert summary_json
    except Exception as e:
        if is_llm_budget_exception(e):
            raise
        if max_retry > 0:
            print(
                "Summary update failed with "
                f"{type(e).__name__}; retrying ({max_retry} attempts left)"
            )
            return update_summary(
                prev_summary,
                cur_stage_name,
                cur_journal,
                cur_summary,
                model,
                client,
                max_retry - 1,
            )
        else:
            print(
                "Summary update failed after bounded retries " f"({type(e).__name__})"
            )
            raise
    return summary_json


overall_plan_summarizer_prompt = """You have been provided with the plans for both the parent node and the current node. Your task is to synthesize a comprehensive summary of the overall plan by integrating details from both the parent and current node plans.
The summary should be thorough and clearly articulate the underlying motivations.
For example, if in your previous overall plan you were experimenting with a new idea, and now your current plan is to fix certain bugs in the previous implementation, your returned overall plan should focus on your previous overall plan, and briefly mention that the current plan includes bug fixes. If your current plan is more about implementing new ideas, then you should summarize that thoroughly along with the previous overall plan.
The goal is to create a comprehensive summary of all historical plans, focusing on the main scientific planning and objectives.

Previous overall plan:
{prev_overall_plan}

Current plan:
{current_plan}

Respond in the following format:

THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```

In <THOUGHT>, thoroughly reason as an expert researcher. First, reason over each node, and then carefully combine all information. It is okay to be very detailed.

In <JSON>, provide the review in JSON format with the following field in exactly this order:
- "overall_plan": a string that describes the overall plan based on the current and previous overall plans

Ensure the JSON is valid and properly formatted, as it will be automatically parsed.
"""


def annotate_history(journal, cfg=None):
    for node in journal.nodes:
        if node.parent:
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    if cfg is None or getattr(cfg, "report", None) is None:
                        raise ValueError("Report model configuration is required")
                    model = cfg.report.model
                    client = get_ai_client(model)
                    response = get_response_from_llm(
                        overall_plan_summarizer_prompt.format(
                            prev_overall_plan=node.parent.overall_plan,
                            current_plan=node.plan,
                        ),
                        client,
                        model,
                        report_summarizer_sys_msg,
                    )
                    node.overall_plan = extract_json_between_markers(response[0])[
                        "overall_plan"
                    ]
                    break
                except Exception as e:
                    if is_llm_budget_exception(e):
                        raise
                    retry_count += 1
                    if retry_count == max_retries:
                        print(
                            f"History annotation failed after {max_retries} attempts "
                            f"({type(e).__name__})"
                        )
                        raise
                    print(
                        "History annotation failed with "
                        f"{type(e).__name__}; retrying "
                        f"({max_retries - retry_count} attempts left)"
                    )
        else:
            node.overall_plan = node.plan


def overall_summarize(journals, cfg=None):
    from concurrent.futures import ThreadPoolExecutor

    if cfg is None or getattr(cfg, "report", None) is None:
        raise ValueError("Final report requires an explicit report model")
    report_model = cfg.report.model
    if not isinstance(report_model, str) or not report_model.strip():
        raise ValueError("Final report model must be a non-empty route")
    journal_items = list(journals)
    if len(journal_items) != 4:
        raise ValueError("Final report requires exactly four qualified stages")

    # Validate the complete ordered provenance set before starting any model
    # call.  Receipt equality prevents relabelling, while the main-stage prefix
    # prevents a valid set of receipts from being silently reordered.
    seen_stage_names: set[str] = set()
    for index, item in enumerate(journal_items, start=1):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Final report contains an invalid stage entry")
        stage_name, journal = item
        prefix, separator, _remainder = str(stage_name).partition("_")
        if (
            not isinstance(stage_name, str)
            or not stage_name
            or stage_name in seen_stage_names
            or not separator
            or not prefix.isdigit()
            or int(prefix) != index
        ):
            raise ValueError("Final report stages are not in canonical order")
        seen_stage_names.add(stage_name)
        evidence = _validated_stage_evidence(
            journal,
            expected_stage_name=stage_name,
        )
        if journal.get_best_node_by_metric() is not evidence["qualified"]:
            raise ValueError("Report journal selected an unqualified node")

    report_client = get_ai_client(report_model)

    def process_stage(idx, stage_tuple):
        stage_name, journal = stage_tuple
        if idx not in range(4):
            raise ValueError("Final report contains an unexpected stage")
        evidence = _validated_stage_evidence(
            journal,
            expected_stage_name=stage_name,
        )
        best_node = journal.get_best_node_by_metric()
        if best_node is not evidence["qualified"]:
            raise ValueError("Report journal selected an unqualified node")
        summary = get_stage_summary(
            journal,
            stage_name,
            report_model,
            report_client,
        )
        # Do not trust a helper return value merely because it is a mapping.
        # This second validation also protects callers that monkey-patch or
        # replace the report writer.
        return _validate_completed_summary(
            summary,
            stage_name=stage_name,
            journal=journal,
        )

    from tqdm import tqdm

    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(
                    process_stage,
                    range(len(journal_items)),
                    journal_items,
                ),
                desc="Processing stages",
                total=len(journal_items),
            )
        )
        if len(results) != 4 or any(result is None for result in results):
            raise ValueError("Final report did not produce four validated summaries")
        draft_summary, baseline_summary, research_summary, ablation_summary = results

    return draft_summary, baseline_summary, research_summary, ablation_summary


if __name__ == "__main__":
    # Test
    example_path = "logs/247-run"

    def load_stage_folders(base_path):
        """
        Load the folders that start with 'stage_' followed by a number.

        Args:
            base_path (str): The base directory path where stage folders are located.

        Returns:
            list: A sorted list of stage folder paths.
        """
        stage_folders = []
        for folder_name in os.listdir(base_path):
            if folder_name.startswith("stage_"):
                stage_folders.append(os.path.join(base_path, folder_name))
        return sorted(stage_folders, key=lambda x: int(x.split("_")[1]))

    def reconstruct_journal(journal_data):
        # Create a mapping of node IDs to Node instances
        id_to_node = {}
        for node_data in journal_data["nodes"]:
            # Remove unused or invalid keys if needed
            if "actionable_insights_from_plots" in node_data:
                del node_data["actionable_insights_from_plots"]
            node = Node.from_dict(node_data)
            id_to_node[node.id] = node

        # Set up parent-child relationships using node2parent
        for node_id, parent_id in journal_data["node2parent"].items():
            child_node = id_to_node[node_id]
            parent_node = id_to_node[parent_id]
            child_node.parent = parent_node
            parent_node.children.add(child_node)

        # Create a Journal and add all nodes
        journal = Journal()
        journal.nodes.extend(id_to_node.values())

        return journal

    # Example usage
    stage_folders = load_stage_folders(example_path)
    journals = []
    for index, folder in enumerate(stage_folders, start=1):
        print(f"Stage {index}: {folder}")
        stage_name = os.path.basename(folder)
        journal_path = os.path.join(folder, "journal.json")
        if os.path.exists(journal_path):
            with open(journal_path, "r") as file:
                journal_data = json.load(file)
                print(f"Loaded journal.json for Stage {index}")
        else:
            print(f"No journal.json found for Stage {index}")
        journal = reconstruct_journal(journal_data)
        journals.append((stage_name, journal))

    # Convert manager journals to list of (stage_name, journal) tuples
    (
        draft_summary,
        baseline_summary,
        research_summary,
        ablation_summary,
    ) = overall_summarize(journals)
    log_dir = "logs/247-run"
    draft_summary_path = log_dir + "/draft_summary.json"
    baseline_summary_path = log_dir + "/baseline_summary.json"
    research_summary_path = log_dir + "/research_summary.json"
    ablation_summary_path = log_dir + "/ablation_summary.json"

    with open(draft_summary_path, "w") as draft_file:
        json.dump(draft_summary, draft_file, indent=2)

    with open(baseline_summary_path, "w") as baseline_file:
        json.dump(baseline_summary, baseline_file, indent=2)

    with open(research_summary_path, "w") as research_file:
        json.dump(research_summary, research_file, indent=2)

    with open(ablation_summary_path, "w") as ablation_file:
        json.dump(ablation_summary, ablation_file, indent=2)

    print(f"Summary reports written to files:")
    print(f"- Draft summary: {draft_summary_path}")
    print(f"- Baseline summary: {baseline_summary_path}")
    print(f"- Research summary: {research_summary_path}")
    print(f"- Ablation summary: {ablation_summary_path}")
