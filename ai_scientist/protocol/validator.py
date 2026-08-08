"""ARA conformance validator.

The published schemas are the protocol contract, so validation uses the full
JSON Schema 2020-12 vocabulary.  In particular, enum, pattern, oneOf, $ref,
uniqueItems, and additionalProperties constraints must not silently disappear
when an artifact crosses an implementation boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "protocol_version": PROTOCOL_VERSION,
            "checked": list(self.checked),
            "errors": [issue.__dict__ for issue in self.errors],
            "warnings": [issue.__dict__ for issue in self.warnings],
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
    return _validate_kind(payload, Kind.MANIFEST, source)


def _iter_json_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))


def validate_ara(ara_root: str | Path, *, strict: bool = False) -> ValidationReport:
    """Full conformance check over an ARA directory.

    Parameters
    ----------
    ara_root:
        Path to the directory containing manifest.json.
    strict:
        When True, treat *warnings* about optional-but-recommended pieces as
        errors. Default False so mildly-lossy exports still validate.
    """
    ara_root = Path(ara_root).expanduser().resolve()
    report = ValidationReport()

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

    if strict:
        for warning in list(report.warnings):
            report.errors.append(
                ValidationIssue(warning.path, warning.message, severity="error")
            )
            report.ok = False
        report.warnings.clear()

    return report
