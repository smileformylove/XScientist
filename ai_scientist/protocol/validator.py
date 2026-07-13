"""ARA conformance validator.

Design choice: we implement a *minimal* JSON Schema checker inline rather than
pulling in the ``jsonschema`` package. Two reasons:

  1. The protocol package should stay dependency-free so anyone can port it.
  2. We only need ``required``, ``type``, ``const``, ``minimum``, and object
     recursion. Full JSON Schema semantics are overkill.

If a downstream consumer wants strict validation with the real library, they
can call ``jsonschema.validate(payload, load_schema(kind))`` themselves — our
schemas ARE valid JSON Schema, just checked with a subset here.
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
from .schemas import load_schema

# Map JSON Schema "type" tokens to Python type checks.
_JSON_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


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


def _type_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_type_matches(value, t) for t in expected)
    checker = _JSON_TYPE_CHECKS.get(expected)
    return bool(checker and checker(value))


def _validate_against_schema(
    payload: Any, schema: dict, path: str, report: ValidationReport
) -> None:
    """Run our minimal-JSON-Schema check.

    Handles: required, type, const, minimum, and recursive object/array
    checks. Silently ignores schema keywords we don't understand (they're
    still legal JSON Schema; we just don't enforce them here).
    """
    if "type" in schema and not _type_matches(payload, schema["type"]):
        report.add_error(path, f"expected type {schema['type']}, got {type(payload).__name__}")
        return  # further checks would compound the noise

    if "const" in schema and payload != schema["const"]:
        report.add_error(path, f"expected const {schema['const']!r}, got {payload!r}")

    if "minimum" in schema and isinstance(payload, (int, float)) and payload < schema["minimum"]:
        report.add_error(path, f"value {payload} < minimum {schema['minimum']}")

    if isinstance(payload, dict):
        for req in schema.get("required", []) or []:
            if req not in payload:
                report.add_error(path or "$", f"missing required field '{req}'")
        for key, sub_schema in (schema.get("properties") or {}).items():
            if key in payload and isinstance(sub_schema, dict):
                _validate_against_schema(
                    payload[key], sub_schema, f"{path}.{key}" if path else key, report
                )

    if isinstance(payload, list) and isinstance(schema.get("items"), dict):
        item_schema = schema["items"]
        for idx, item in enumerate(payload):
            _validate_against_schema(item, item_schema, f"{path}[{idx}]", report)


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


def validate_manifest(payload: dict, *, source: str = "manifest.json") -> ValidationReport:
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
            report.merge(_validate_kind(graph_payload, Kind.EXPLORATION_GRAPH, "exploration_graph.json"))
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
                    message = str(issue.get("message") or issue.get("code") or "graph issue")
                    if issue.get("severity") == "warning":
                        report.add_warning(path, message)
                    else:
                        report.add_error(path, message)

    # Optional folders — only validate contents if present.
    if isinstance(graph_payload, dict):
        node_ids = {
            str(n.get("id")) for n in (graph_payload.get("nodes") or []) if isinstance(n, dict)
        }
        nodes_dir = ara_root / "nodes"
        if nodes_dir.exists():
            for node_id in sorted(node_ids):
                node_dir = nodes_dir / node_id
                if not node_dir.exists():
                    report.add_warning(f"nodes/{node_id}", "listed in graph but missing on disk")
                    continue
                metrics_path = node_dir / "metrics.json"
                if metrics_path.exists():
                    try:
                        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        report.add_error(f"nodes/{node_id}/metrics.json", f"unreadable: {exc}")
                        continue
                    report.merge(_validate_kind(metrics, Kind.NODE, f"nodes/{node_id}/metrics.json"))

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
        kind = Kind.REEXEC_BATCH if schema_tag == "ara.reexec.batch.v1" else Kind.VERIFY_REPORT
        report.merge(_validate_kind(payload, kind, f"verify/{verify_file.name}"))

    if strict:
        for warning in list(report.warnings):
            report.errors.append(
                ValidationIssue(warning.path, warning.message, severity="error")
            )
            report.ok = False
        report.warnings.clear()

    return report
