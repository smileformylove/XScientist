"""ARA conformance validator.

The published schemas are the protocol contract, so validation uses the full
JSON Schema 2020-12 vocabulary.  In particular, enum, pattern, oneOf, $ref,
uniqueItems, and additionalProperties constraints must not silently disappear
when an artifact crosses an implementation boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .constants import (
    PROTOCOL_VERSION,
    REQUIRED_TOP_LEVEL,
    Kind,
)
from .graph import analyze_exploration_graph
from .schemas import load_schema, schema_registry


@dataclass
class ValidationIssue:
    path: str
    message: str
    severity: str = "error"  # "error" | "warning"

    def format(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.severity}] {self.path}: {self.message}"


@dataclass
class ValidationReport:
    ok: bool = True
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    conformance: dict[str, Any] = field(default_factory=dict)

    def add_error(self, path: str, message: str) -> None:
        self.ok = False
        self.errors.append(ValidationIssue(path, message, "error"))

    def add_warning(self, path: str, message: str) -> None:
        self.warnings.append(ValidationIssue(path, message, "warning"))

    def merge(self, other: "ValidationReport") -> None:
        if not other.ok:
            self.ok = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.checked.extend(other.checked)
        if other.conformance:
            self.conformance.update(other.conformance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "protocol_version": PROTOCOL_VERSION,
            "checked": list(self.checked),
            "errors": [issue.__dict__ for issue in self.errors],
            "warnings": [issue.__dict__ for issue in self.warnings],
            "conformance": dict(self.conformance),
        }


PORTABILITY_PROFILE = "ara.portable.v1"
ARA_CONFORMANCE_LEVELS = ("index", "trace", "replay", "verify")


def _is_absolute_portable_path(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and (
        PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute()
    )


def _portable_path_issues(payload: dict[str, Any]) -> list[str]:
    values: list[tuple[str, Any]] = [
        ("source_exp_dir", payload.get("source_exp_dir")),
        ("project_dir", payload.get("project_dir")),
    ]
    provenance = payload.get("provenance") or {}
    if isinstance(provenance, dict):
        values.append(("provenance.parent_ara_root", provenance.get("parent_ara_root")))
        for index, parent in enumerate(provenance.get("parents") or []):
            if isinstance(parent, dict):
                values.append(
                    (
                        f"provenance.parents[{index}].parent_ara_root",
                        parent.get("parent_ara_root"),
                    )
                )
    return [name for name, value in values if _is_absolute_portable_path(value)]


def _assess_conformance(
    ara_root: Path,
    manifest: dict[str, Any] | None,
    graph: dict[str, Any] | None,
) -> dict[str, Any]:
    blockers: dict[str, list[str]] = {level: [] for level in ARA_CONFORMANCE_LEVELS}
    if not isinstance(manifest, dict) or not isinstance(graph, dict):
        blockers["index"].append("manifest_and_graph_required")
    nodes = [
        item for item in (graph or {}).get("nodes") or [] if isinstance(item, dict)
    ]
    for node in nodes:
        node_id = str(node.get("id") or "unknown")
        if not node.get("content_hash"):
            blockers["trace"].append(f"{node_id}:content_hash_missing")
        if not node.get("identity_profile"):
            blockers["trace"].append(f"{node_id}:identity_profile_legacy_or_missing")
        node_dir = ara_root / "nodes" / node_id
        if not (node_dir / "metrics.json").exists():
            blockers["replay"].append(f"{node_id}:metrics_missing")
        if not (node_dir / "code.py").exists():
            blockers["replay"].append(f"{node_id}:code_missing")
        if not (node_dir / "run.sh").exists():
            blockers["replay"].append(f"{node_id}:runner_missing")
        if not (node_dir / "env.json").exists():
            blockers["replay"].append(f"{node_id}:environment_missing")
    verification_targets: set[str] = set()
    for verify_file in _iter_json_files(ara_root / "verify"):
        try:
            row = json.loads(verify_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("node_id") or "")
        passed = row.get("verdict") == "passed" or row.get("ok") is True
        if node_id and passed:
            verification_targets.add(node_id)
    for node in nodes:
        node_id = str(node.get("id") or "unknown")
        if node.get("is_buggy") is not True and node_id not in verification_targets:
            blockers["verify"].append(f"{node_id}:passing_verification_missing")
    inherited: list[str] = []
    achieved = "none"
    level_results: dict[str, dict[str, Any]] = {}
    for level in ARA_CONFORMANCE_LEVELS:
        inherited.extend(blockers[level])
        level_results[level] = {
            "ok": not inherited,
            "blockers": list(inherited),
        }
        if not inherited:
            achieved = level
    return {
        "profile": "ara.conformance.v1",
        "achieved": achieved,
        "levels": level_results,
    }


def _validate_against_schema(
    payload: Any, schema: dict, path: str, report: ValidationReport
) -> None:
    """Validate one payload using the complete Draft 2020-12 contract."""

    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema, registry=schema_registry())
    for error in sorted(
        validator.iter_errors(payload),
        key=lambda item: (list(item.absolute_path), item.message),
    ):
        suffix = "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error.absolute_path
        )
        location = f"{path}{suffix}" if path else (suffix.lstrip(".") or "$")
        report.add_error(location, error.message)


def _validate_kind(payload: Any, kind: Kind, source: str) -> ValidationReport:
    report = ValidationReport()
    report.checked.append(f"{kind.value}:{source}")
    try:
        schema = load_schema(kind)
    except FileNotFoundError as exc:
        report.add_error(source, f"no schema registered for {kind.value}: {exc}")
        return report
    _validate_against_schema(payload, schema, path="", report=report)
    return report


def validate_manifest(
    payload: dict, *, source: str = "manifest.json"
) -> ValidationReport:
    """Validate a single manifest.json payload."""
    report = _validate_kind(payload, Kind.MANIFEST, source)
    if payload.get("portability_profile") == PORTABILITY_PROFILE:
        for path in _portable_path_issues(payload):
            report.add_error(path, "portable ARA paths must be relative")
    return report


def _iter_json_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))


def validate_ara(
    ara_root: str | Path,
    *,
    strict: bool = False,
    level: str = "index",
) -> ValidationReport:
    """Full conformance check over an ARA directory.

    Parameters
    ----------
    ara_root:
        Path to the directory containing manifest.json.
    strict:
        When True, treat *warnings* about optional-but-recommended pieces as
        errors. Default False so mildly-lossy exports still validate.
    """
    if level not in ARA_CONFORMANCE_LEVELS:
        raise ValueError(f"unsupported ARA conformance level: {level}")
    ara_root = Path(ara_root).expanduser().resolve()
    report = ValidationReport()
    manifest: dict[str, Any] | None = None

    if not ara_root.exists() or not ara_root.is_dir():
        report.add_error(str(ara_root), "ARA root does not exist or is not a directory")
        return report

    for required_name in REQUIRED_TOP_LEVEL:
        target = ara_root / required_name
        if not target.exists():
            report.add_error(required_name, "required file missing")

    manifest_path = ara_root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.add_error("manifest.json", f"unreadable JSON: {exc}")
            manifest = None
        if isinstance(manifest, dict):
            report.merge(validate_manifest(manifest, source="manifest.json"))
            if manifest.get("schema_version") != PROTOCOL_VERSION:
                report.add_warning(
                    "manifest.json.schema_version",
                    f"expected {PROTOCOL_VERSION!r}, saw {manifest.get('schema_version')!r}",
                )

    graph_path = ara_root / "exploration_graph.json"
    graph_payload = None
    if graph_path.exists():
        try:
            graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.add_error("exploration_graph.json", f"unreadable JSON: {exc}")
        if isinstance(graph_payload, dict):
            report.merge(
                _validate_kind(
                    graph_payload, Kind.EXPLORATION_GRAPH, "exploration_graph.json"
                )
            )
            try:
                dag = analyze_exploration_graph(graph_payload)
            except Exception as exc:  # pragma: no cover - defensive
                report.add_error(
                    "exploration_graph.json",
                    f"DAG analysis failed: {exc}",
                )
            else:
                for issue in dag.get("issues") or []:
                    path = str(issue.get("path") or "exploration_graph.json")
                    message = str(
                        issue.get("message") or issue.get("code") or "graph issue"
                    )
                    if issue.get("severity") == "warning":
                        report.add_warning(path, message)
                    else:
                        report.add_error(path, message)

    # Optional folders — only validate contents if present.
    if isinstance(graph_payload, dict):
        node_ids = {
            str(n.get("id"))
            for n in (graph_payload.get("nodes") or [])
            if isinstance(n, dict)
        }
        nodes_dir = ara_root / "nodes"
        if nodes_dir.exists():
            for node_id in sorted(node_ids):
                node_dir = nodes_dir / node_id
                if not node_dir.exists():
                    report.add_warning(
                        f"nodes/{node_id}", "listed in graph but missing on disk"
                    )
                    continue
                metrics_path = node_dir / "metrics.json"
                if metrics_path.exists():
                    try:
                        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        report.add_error(
                            f"nodes/{node_id}/metrics.json", f"unreadable: {exc}"
                        )
                        continue
                    report.merge(
                        _validate_kind(
                            metrics, Kind.NODE, f"nodes/{node_id}/metrics.json"
                        )
                    )

    claims_dir = ara_root / "claims"
    for claim_file in _iter_json_files(claims_dir):
        if claim_file.name.startswith("_"):
            continue  # _index.json is a summary, not an individual claim
        try:
            claim = json.loads(claim_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.add_error(f"claims/{claim_file.name}", f"unreadable: {exc}")
            continue
        report.merge(_validate_kind(claim, Kind.CLAIM, f"claims/{claim_file.name}"))

    verify_dir = ara_root / "verify"
    for verify_file in _iter_json_files(verify_dir):
        try:
            payload = json.loads(verify_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        schema_tag = payload.get("schema") if isinstance(payload, dict) else None
        kind = (
            Kind.REEXEC_BATCH
            if schema_tag == "ara.reexec.batch.v1"
            else Kind.VERIFY_REPORT
        )
        report.merge(_validate_kind(payload, kind, f"verify/{verify_file.name}"))

    report.conformance = _assess_conformance(ara_root, manifest, graph_payload)
    requested_index = ARA_CONFORMANCE_LEVELS.index(level)
    achieved = str(report.conformance.get("achieved") or "none")
    achieved_index = (
        ARA_CONFORMANCE_LEVELS.index(achieved)
        if achieved in ARA_CONFORMANCE_LEVELS
        else -1
    )
    if achieved_index < requested_index:
        report.add_error(
            "conformance",
            f"requested {level} but artifact achieved {achieved}",
        )

    if strict:
        for warning in list(report.warnings):
            report.errors.append(
                ValidationIssue(warning.path, warning.message, severity="error")
            )
            report.ok = False
        report.warnings.clear()

    return report
