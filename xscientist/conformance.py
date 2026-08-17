"""Portable producer conformance fixtures for Research VCS objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_scientist.protocol.research_vcs import (
    ResearchObjectError,
    build_research_object,
    validate_research_object,
)
from ai_scientist.protocol.schemas import available_schemas, schema_validator
from ai_scientist.utils.atomic_io import atomic_write_json, atomic_write_text

CONFORMANCE_SCHEMA = "xscientist.protocol-conformance-kit.v1"


def init_conformance_kit(directory: str | Path) -> dict[str, Any]:
    """Create one known-good and one known-bad versioned fixture."""

    root = Path(directory).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("conformance directory must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    good = build_research_object(
        kind="hypothesis",
        payload={
            "statement": "The registered intervention changes the primary metric."
        },
        created_at="2026-01-01T00:00:00+00:00",
    )
    bad = json.loads(json.dumps(good))
    bad["payload"]["statement"] = "Tampered after content addressing."
    atomic_write_json(root / "known-good.research-object.json", good)
    atomic_write_json(root / "known-bad.research-object.json", bad)
    manifest = {
        "schema": CONFORMANCE_SCHEMA,
        "cases": [
            {
                "file": "known-good.research-object.json",
                "schema_name": "research_object",
                "expected_valid": True,
            },
            {
                "file": "known-bad.research-object.json",
                "schema_name": "research_object",
                "expected_valid": False,
            },
        ],
    }
    atomic_write_json(root / "conformance.json", manifest)
    atomic_write_text(
        root / "README.md",
        "# XScientist protocol conformance kit\n\n"
        "Run `xscientist conformance check .` after replacing or adding producer fixtures.\n"
        "Schema validation is offline; Research Objects also receive canonical identity checks.\n",
    )
    return {
        "schema": CONFORMANCE_SCHEMA,
        "ok": True,
        "directory": root.name,
        "cases": len(manifest["cases"]),
        "next_command": f"xscientist conformance check {root.name}",
    }


def _validate_case(path: Path, schema_name: str) -> tuple[bool, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema_validator(schema_name).validate(payload)
        if schema_name == "research_object":
            validate_research_object(payload)
    except (OSError, json.JSONDecodeError, ResearchObjectError, ValueError) as exc:
        return False, str(exc)
    except Exception as exc:  # jsonschema/reference errors are report data here
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def check_conformance(
    target: str | Path,
    *,
    schema_name: str = "research_object",
) -> dict[str, Any]:
    """Check a generated kit directory or one JSON protocol artifact."""

    path = Path(target).expanduser().resolve()
    if schema_name not in available_schemas():
        raise ValueError(f"unknown protocol schema: {schema_name}")
    cases: list[dict[str, Any]] = []
    if path.is_dir():
        manifest_path = path / "conformance.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read conformance.json: {exc}") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != CONFORMANCE_SCHEMA
        ):
            raise ValueError("unsupported conformance manifest")
        raw_cases = manifest.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("conformance manifest has no cases")
        for item in raw_cases:
            if not isinstance(item, dict):
                raise ValueError("conformance case must be an object")
            relative = Path(str(item.get("file") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("conformance case path must stay inside the kit")
            case_schema = str(item.get("schema_name") or schema_name)
            if case_schema not in available_schemas():
                raise ValueError(f"unknown protocol schema: {case_schema}")
            valid, error = _validate_case(path / relative, case_schema)
            expected = item.get("expected_valid") is True
            cases.append(
                {
                    "file": relative.as_posix(),
                    "schema_name": case_schema,
                    "expected_valid": expected,
                    "actual_valid": valid,
                    "passed": valid is expected,
                    "error": error,
                }
            )
    elif path.is_file():
        valid, error = _validate_case(path, schema_name)
        cases.append(
            {
                "file": path.name,
                "schema_name": schema_name,
                "expected_valid": True,
                "actual_valid": valid,
                "passed": valid,
                "error": error,
            }
        )
    else:
        raise ValueError("conformance target does not exist")
    return {
        "schema": "xscientist.protocol-conformance-report.v1",
        "ok": all(bool(case["passed"]) for case in cases),
        "target": path.name,
        "cases": cases,
        "passed": sum(bool(case["passed"]) for case in cases),
        "total": len(cases),
    }


__all__ = ["CONFORMANCE_SCHEMA", "check_conformance", "init_conformance_kit"]
