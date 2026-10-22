"""Fail-closed claim-centric projection for Research DAG exports.

The argument view is an advisory sidecar.  It deliberately does not add nodes,
edges, sources, or fields to the authoritative Research DAG hash scope.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from jsonschema import ValidationError, validate as validate_json

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.hashing import content_hash as protocol_content_hash
from ai_scientist.protocol.research_vcs import (
    ResearchObjectError,
    validate_research_object,
)
from ai_scientist.protocol.schemas import load_schema

from .research_closure import (
    RESEARCH_CLOSURE_SCHEMA,
    _claim_alias_ids,
    _effective_claim_frontier,
    _effective_superseded_ids,
    _is_active_closure_object,
)
from .research_git import ResearchGitError, _repository_root

RESEARCH_ARGUMENT_SCHEMA = "xscientist.research-argument-view.v1"
ADVISORY_ROUTE_INDEX_SCHEMA = "xscientist.advisory-route-index.v1"
ARA_CLAIM_ROLE_BINDING_SCHEMA = "xscientist.ara-claim-role-binding.v1"
_RESEARCH_DAG_SCHEMA = "xscientist.research-dag.v1"

_RAW_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CONTENT_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_RSO_ID_RE = re.compile(r"rso-[0-9a-f]{16}")
_ACTIVE_EVIDENCE_STATES = {"completed", "verified", "promoted"}
_DIRECT_SEMANTIC_CATEGORIES = {"argument", "challenge", "support"}
_PROVENANCE_CATEGORIES = {"context", "evolution", "lineage", "provenance"}
_SUPPORT_RELATION_TYPES = {"supports", "qualified_supports"}
_CHALLENGE_RELATION_TYPES = {
    "refutes",
    "qualified_refutes",
    "contradicts",
    "challenges_inference",
    "invalidates",
}
_ROLE_EDGE_CATEGORY = {
    "support": "support",
    "challenge": "challenge",
    "boundary": "argument",
    "control": "argument",
}
_ROLE_EDGE_TYPES = {
    "support": _SUPPORT_RELATION_TYPES,
    "challenge": _CHALLENGE_RELATION_TYPES,
    "boundary": {"qualifies"},
    "control": {"uses_method"},
}

_MAX_ADVISORY_INDEX_BYTES = 1024 * 1024
_MAX_ADVISORY_FILE_BYTES = 16 * 1024 * 1024
_MAX_ADVISORY_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_ADVISORY_FILES = 256
_MAX_ADVISORY_ROUTES = 64
_MAX_ROUTE_ARTIFACTS = 64
_MAX_ROUTE_GATES = 128
_MAX_ROUTE_FINDINGS = 128
_MAX_SHORT_TEXT = 256
_MAX_LONG_TEXT = 8192
_MAX_SAFE_INTEGER = (1 << 53) - 1


class ResearchArgumentError(ResearchGitError):
    """Raised when an argument sidecar cannot be projected safely."""


def _open_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode)


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _secure_read_capabilities_available() -> bool:
    return bool(
        os.name == "posix"
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
    )


def _open_repository_root(repo: str | Path) -> tuple[Path, int]:
    """Open the resolved repository root as the only filesystem trust anchor."""

    if not _secure_read_capabilities_available():
        raise ResearchArgumentError(
            "secure descriptor-relative advisory reads are unavailable"
        )
    root = _repository_root(repo)
    try:
        before = root.lstat()
    except OSError as exc:
        raise ResearchArgumentError(
            f"research repository is unavailable: {root}"
        ) from exc
    if not stat.S_ISDIR(before.st_mode):
        raise ResearchArgumentError(f"research repository is not a directory: {root}")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
        after = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ResearchArgumentError(
            f"cannot securely open research repository: {root}"
        ) from exc
    assert descriptor is not None
    if _open_identity(before) != _open_identity(after):
        os.close(descriptor)
        raise ResearchArgumentError(
            f"research repository changed while opening: {root}"
        )
    return root, descriptor


def _relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ResearchArgumentError(f"{label} must be a non-empty relative path")
    if "\\" in value or "\x00" in value:
        raise ResearchArgumentError(
            f"{label} is not a normalized POSIX path: {value!r}"
        )
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not raw_parts
        or any(part in {"", ".", ".."} for part in raw_parts)
        or path.as_posix() != value
    ):
        raise ResearchArgumentError(f"{label} is not a safe relative path: {value!r}")
    return value


def _index_relative_path(root: Path, value: str | Path) -> str:
    supplied = Path(value).expanduser()
    if supplied.is_absolute():
        try:
            relative = supplied.relative_to(root)
        except ValueError:
            # macOS exposes the same trusted directory through aliases such as
            # /var and /private/var.  Match an ancestor to the already-opened
            # physical root, but retain every descendant component so the
            # descriptor walk below still observes and rejects symlinks.
            relative = None
            for ancestor in supplied.parents:
                try:
                    same_root = ancestor.resolve(strict=True) == root
                except OSError:
                    continue
                if same_root:
                    relative = supplied.relative_to(ancestor)
                    break
            if relative is None:
                raise ResearchArgumentError(
                    "advisory index must be located inside the research repository"
                )
        candidate = relative.as_posix()
    else:
        candidate = os.fspath(supplied)
    return _relative_path(candidate, label="advisory index path")


def _read_relative_file(
    root_descriptor: int,
    relative_path: str,
    *,
    maximum: int,
    label: str,
    missing_ok: bool,
) -> bytes | None:
    """Read stable bytes without following a symlink in any path component."""

    normalized = _relative_path(relative_path, label=f"{label} path")
    if not isinstance(maximum, int) or maximum < 0:
        raise ResearchArgumentError(f"invalid {label} byte limit")
    directory_descriptor = os.dup(root_descriptor)
    file_descriptor: int | None = None
    try:
        directory_flags = (
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        )
        for component in PurePosixPath(normalized).parts[:-1]:
            child_descriptor: int | None = None
            try:
                before = os.stat(
                    component,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if missing_ok:
                    return None
                raise ResearchArgumentError(f"{label} is missing: {normalized}")
            if not stat.S_ISDIR(before.st_mode):
                raise ResearchArgumentError(
                    f"{label} ancestor is not a real directory: {normalized}"
                )
            try:
                child_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
                after = os.fstat(child_descriptor)
            except OSError as exc:
                if child_descriptor is not None:
                    os.close(child_descriptor)
                raise ResearchArgumentError(
                    f"cannot securely open {label} ancestor: {normalized}"
                ) from exc
            if _open_identity(before) != _open_identity(after):
                os.close(child_descriptor)
                raise ResearchArgumentError(
                    f"{label} ancestor changed while opening: {normalized}"
                )
            os.close(directory_descriptor)
            directory_descriptor = child_descriptor

        leaf = PurePosixPath(normalized).parts[-1]
        try:
            before = os.stat(
                leaf,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            if missing_ok:
                return None
            raise ResearchArgumentError(f"{label} is missing: {normalized}")
        if stat.S_ISLNK(before.st_mode):
            raise ResearchArgumentError(f"{label} symlink is forbidden: {normalized}")
        if not stat.S_ISREG(before.st_mode):
            raise ResearchArgumentError(f"{label} is not a regular file: {normalized}")
        if before.st_nlink != 1:
            raise ResearchArgumentError(f"{label} hardlink is forbidden: {normalized}")
        if before.st_size > maximum:
            raise ResearchArgumentError(
                f"{label} exceeds {maximum} bytes: {normalized}"
            )
        try:
            file_descriptor = os.open(
                leaf,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_descriptor,
            )
            opened = os.fstat(file_descriptor)
        except OSError as exc:
            raise ResearchArgumentError(
                f"cannot securely open {label}: {normalized}"
            ) from exc
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _stable_identity(before) != _stable_identity(opened)
        ):
            raise ResearchArgumentError(f"{label} changed while opening: {normalized}")

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_descriptor, min(65_536, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise ResearchArgumentError(
                    f"{label} exceeds {maximum} bytes: {normalized}"
                )
        finished = os.fstat(file_descriptor)
        if (
            _stable_identity(opened) != _stable_identity(finished)
            or total != finished.st_size
        ):
            raise ResearchArgumentError(f"{label} changed while reading: {normalized}")
        return b"".join(chunks)
    except ResearchArgumentError:
        raise
    except OSError as exc:
        raise ResearchArgumentError(
            f"cannot securely read {label}: {normalized}"
        ) from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(directory_descriptor)


def _strict_json(encoded: bytes, *, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return value

    def safe_int(token: str) -> int:
        value = int(token)
        if abs(value) > _MAX_SAFE_INTEGER:
            raise ValueError("JSON integer exceeds interoperable safe range")
        return value

    def reject_constant(token: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {token}")

    try:
        decoded = encoded.decode("utf-8", errors="strict")
        return json.loads(
            decoded,
            object_pairs_hook=unique_object,
            parse_float=finite_float,
            parse_int=safe_int,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResearchArgumentError(f"{label} is not strict JSON: {exc}") from exc


def _exact_object(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResearchArgumentError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ResearchArgumentError(
            f"{label} has invalid keys (missing={missing}, extra={extra})"
        )
    return value


def _bounded_text(
    value: Any,
    *,
    label: str,
    maximum: int = _MAX_LONG_TEXT,
    allowed: set[str] | None = None,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
        or "\x00" in value
    ):
        raise ResearchArgumentError(f"{label} must be bounded, non-empty text")
    if allowed is not None and value not in allowed:
        raise ResearchArgumentError(f"{label} has unsupported value: {value}")
    return value


def _raw_hash(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _RAW_SHA256_RE.fullmatch(value) is None:
        raise ResearchArgumentError(
            f"{label} must be a bare lowercase 64-character SHA-256 digest"
        )
    return value


def _binding(value: Any, *, label: str) -> dict[str, str]:
    row = _exact_object(value, {"path", "raw_sha256"}, label=label)
    return {
        "path": _relative_path(row["path"], label=f"{label}.path"),
        "raw_sha256": _raw_hash(row["raw_sha256"], label=f"{label}.raw_sha256"),
    }


def _validate_advisory_index(value: Any) -> dict[str, Any]:
    index = _exact_object(value, {"schema_version", "routes"}, label="advisory index")
    if index["schema_version"] != ADVISORY_ROUTE_INDEX_SCHEMA:
        raise ResearchArgumentError("unsupported advisory index schema_version")
    routes = index["routes"]
    if not isinstance(routes, list) or len(routes) > _MAX_ADVISORY_ROUTES:
        raise ResearchArgumentError(
            f"advisory index routes must contain at most {_MAX_ADVISORY_ROUTES} items"
        )
    route_ids: set[str] = set()
    normalized_routes: list[dict[str, Any]] = []
    route_keys = {
        "route_id",
        "question",
        "authority",
        "artifact_bindings",
        "gate_contract_binding",
        "gate_results",
        "open_findings",
        "novelty_disposition",
        "registration_state",
        "permitted_next_action",
    }
    for index_number, raw_route in enumerate(routes):
        label = f"advisory route[{index_number}]"
        route = _exact_object(raw_route, route_keys, label=label)
        route_id = _bounded_text(
            route["route_id"], label=f"{label}.route_id", maximum=_MAX_SHORT_TEXT
        )
        if route_id in route_ids:
            raise ResearchArgumentError(f"duplicate advisory route_id: {route_id}")
        route_ids.add(route_id)
        authority = _bounded_text(
            route["authority"],
            label=f"{label}.authority",
            allowed={"UNREGISTERED"},
        )
        registration_state = _bounded_text(
            route["registration_state"],
            label=f"{label}.registration_state",
            allowed={"absent"},
        )

        raw_artifacts = route["artifact_bindings"]
        if (
            not isinstance(raw_artifacts, list)
            or not raw_artifacts
            or len(raw_artifacts) > _MAX_ROUTE_ARTIFACTS
        ):
            raise ResearchArgumentError(
                f"{label}.artifact_bindings must contain 1-{_MAX_ROUTE_ARTIFACTS} items"
            )
        artifacts = [
            _binding(item, label=f"{label}.artifact_bindings[{item_index}]")
            for item_index, item in enumerate(raw_artifacts)
        ]
        artifact_paths = [item["path"] for item in artifacts]
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ResearchArgumentError(f"{label} contains duplicate artifact paths")

        raw_gates = route["gate_results"]
        if not isinstance(raw_gates, list) or len(raw_gates) > _MAX_ROUTE_GATES:
            raise ResearchArgumentError(
                f"{label}.gate_results must contain at most {_MAX_ROUTE_GATES} items"
            )
        gates: list[dict[str, str]] = []
        gate_ids: set[str] = set()
        for gate_index, raw_gate in enumerate(raw_gates):
            gate_label = f"{label}.gate_results[{gate_index}]"
            gate = _exact_object(raw_gate, {"gate_id", "status"}, label=gate_label)
            gate_id = _bounded_text(
                gate["gate_id"], label=f"{gate_label}.gate_id", maximum=_MAX_SHORT_TEXT
            )
            if gate_id in gate_ids:
                raise ResearchArgumentError(
                    f"{label} contains duplicate gate_id: {gate_id}"
                )
            gate_ids.add(gate_id)
            gates.append(
                {
                    "gate_id": gate_id,
                    "status": _bounded_text(
                        gate["status"],
                        label=f"{gate_label}.status",
                        allowed={"PASS", "FAIL", "MIXED", "UNREVIEWED"},
                    ),
                }
            )

        raw_findings = route["open_findings"]
        if (
            not isinstance(raw_findings, list)
            or len(raw_findings) > _MAX_ROUTE_FINDINGS
        ):
            raise ResearchArgumentError(
                f"{label}.open_findings must contain at most {_MAX_ROUTE_FINDINGS} items"
            )
        findings: list[dict[str, str]] = []
        for finding_index, raw_finding in enumerate(raw_findings):
            finding_label = f"{label}.open_findings[{finding_index}]"
            finding = _exact_object(
                raw_finding, {"severity", "statement"}, label=finding_label
            )
            findings.append(
                {
                    "severity": _bounded_text(
                        finding["severity"],
                        label=f"{finding_label}.severity",
                        allowed={"P0", "P1", "P2", "P3"},
                    ),
                    "statement": _bounded_text(
                        finding["statement"], label=f"{finding_label}.statement"
                    ),
                }
            )

        normalized_routes.append(
            {
                "route_id": route_id,
                "question": _bounded_text(route["question"], label=f"{label}.question"),
                "authority": authority,
                "artifact_bindings": artifacts,
                "gate_contract_binding": _binding(
                    route["gate_contract_binding"],
                    label=f"{label}.gate_contract_binding",
                ),
                "gate_results": gates,
                "open_findings": findings,
                "novelty_disposition": _bounded_text(
                    route["novelty_disposition"],
                    label=f"{label}.novelty_disposition",
                ),
                "registration_state": registration_state,
                "permitted_next_action": _bounded_text(
                    route["permitted_next_action"],
                    label=f"{label}.permitted_next_action",
                ),
            }
        )
    return {"schema_version": ADVISORY_ROUTE_INDEX_SCHEMA, "routes": normalized_routes}


def _load_advisory_index_from_descriptor(
    root: Path,
    root_descriptor: int,
    index_path: str | Path,
) -> tuple[str, bytes, dict[str, Any]]:
    relative = _index_relative_path(root, index_path)
    encoded = _read_relative_file(
        root_descriptor,
        relative,
        maximum=_MAX_ADVISORY_INDEX_BYTES,
        label="advisory index",
        missing_ok=False,
    )
    assert encoded is not None
    return (
        relative,
        encoded,
        _validate_advisory_index(_strict_json(encoded, label="advisory index")),
    )


def load_advisory_route_index(
    repo: str | Path, index_path: str | Path
) -> dict[str, Any]:
    """Load and validate one explicit advisory index without scanning the repo."""

    root, root_descriptor = _open_repository_root(repo)
    try:
        _relative, _encoded, index = _load_advisory_index_from_descriptor(
            root, root_descriptor, index_path
        )
        return index
    finally:
        os.close(root_descriptor)


def project_advisory_route_index(
    repo: str | Path,
    index_path: str | Path,
    *,
    disclose_summaries: bool = True,
) -> dict[str, Any]:
    """Verify advisory bytes and return separately labelled route projections."""

    root, root_descriptor = _open_repository_root(repo)
    try:
        relative, encoded, index = _load_advisory_index_from_descriptor(
            root, root_descriptor, index_path
        )
        cache: dict[str, bytes | None] = {}
        total_bytes = len(encoded)
        if total_bytes > _MAX_ADVISORY_TOTAL_BYTES:
            raise ResearchArgumentError("advisory inputs exceed aggregate byte budget")
        projected: list[dict[str, Any]] = []
        for route in index["routes"]:
            failures: list[dict[str, Any]] = []
            declared_bindings = [
                *route["artifact_bindings"],
                route["gate_contract_binding"],
            ]
            bindings = list(
                {
                    (binding["path"], binding["raw_sha256"]): binding
                    for binding in declared_bindings
                }.values()
            )
            for binding in bindings:
                path = binding["path"]
                if path not in cache:
                    if len(cache) >= _MAX_ADVISORY_FILES:
                        raise ResearchArgumentError(
                            f"advisory index exceeds {_MAX_ADVISORY_FILES} unique files"
                        )
                    cache[path] = _read_relative_file(
                        root_descriptor,
                        path,
                        maximum=_MAX_ADVISORY_FILE_BYTES,
                        label="advisory artifact",
                        missing_ok=True,
                    )
                    if cache[path] is not None:
                        total_bytes += len(cache[path] or b"")
                        if total_bytes > _MAX_ADVISORY_TOTAL_BYTES:
                            raise ResearchArgumentError(
                                "advisory inputs exceed aggregate byte budget"
                            )
                artifact_bytes = cache[path]
                observed = (
                    hashlib.sha256(artifact_bytes).hexdigest()
                    if artifact_bytes is not None
                    else None
                )
                if observed != binding["raw_sha256"]:
                    failures.append(
                        {
                            "path": path,
                            "expected_raw_sha256": binding["raw_sha256"],
                            "observed_raw_sha256": observed,
                            "reason": (
                                "MISSING" if observed is None else "HASH_MISMATCH"
                            ),
                        }
                    )
            if failures:
                projected.append(
                    {
                        "route_id": route["route_id"],
                        "authority": route["authority"],
                        "registration_state": route["registration_state"],
                        "binding_state": "HASH_MISMATCH",
                        "route_state": "BLOCKED",
                        "binding_failures": failures,
                    }
                )
                continue
            projected_route = {
                **route,
                "binding_state": "VERIFIED",
                "route_state": "ADVISORY",
            }
            if not disclose_summaries:
                projected_route = {
                    key: projected_route[key]
                    for key in (
                        "route_id",
                        "authority",
                        "artifact_bindings",
                        "gate_contract_binding",
                        "registration_state",
                        "binding_state",
                        "route_state",
                    )
                }
            projected.append(projected_route)
        return {
            "index_binding": {
                "schema_version": ADVISORY_ROUTE_INDEX_SCHEMA,
                "path": relative,
                "raw_sha256": hashlib.sha256(encoded).hexdigest(),
            },
            "routes": projected,
        }
    finally:
        os.close(root_descriptor)


def _content_hash(value: Any) -> bool:
    return isinstance(value, str) and _CONTENT_SHA256_RE.fullmatch(value) is not None


def _validated_graph_collections(
    graph: Mapping[str, Any],
) -> tuple[list[Any], list[Any]]:
    """Validate the immutable DAG binding used by every argument projection."""

    if graph.get("schema_version") != _RESEARCH_DAG_SCHEMA:
        raise ResearchArgumentError("unsupported research DAG schema_version")
    if not isinstance(graph.get("ref"), str) or not graph["ref"]:
        raise ResearchArgumentError("research DAG ref is invalid")
    if not isinstance(graph.get("commit"), str) or not graph["commit"]:
        raise ResearchArgumentError("research DAG commit is invalid")
    if not isinstance(graph.get("content_disclosed"), bool):
        raise ResearchArgumentError("research DAG content_disclosed is invalid")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ResearchArgumentError("research DAG nodes and edges must be arrays")
    if any(not isinstance(node, Mapping) for node in nodes):
        raise ResearchArgumentError("research DAG node must be an object")
    if any(not isinstance(edge, Mapping) for edge in edges):
        raise ResearchArgumentError("research DAG edge must be an object")
    node_ids = [str(node.get("id") or "") for node in nodes]
    if any(not node_id for node_id in node_ids) or len(node_ids) != len(set(node_ids)):
        raise ResearchArgumentError("research DAG node identities are invalid")
    declared_hash = graph.get("graph_hash")
    if not _content_hash(declared_hash):
        raise ResearchArgumentError("research DAG graph_hash is invalid")
    graph_core = {
        key: value
        for key, value in graph.items()
        if key not in {"generated_at", "graph_hash"}
    }
    try:
        expected_hash = canonical_content_hash(graph_core)
    except (TypeError, ValueError) as exc:
        raise ResearchArgumentError(
            "research DAG hash scope is not canonical JSON"
        ) from exc
    if declared_hash != expected_hash:
        raise ResearchArgumentError("research DAG graph_hash mismatch")
    return nodes, edges


def _validated_research_objects(
    graph: Mapping[str, Any], object_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    """Bind every Research VCS row to its exact authoritative DAG node."""

    if isinstance(object_rows, (str, bytes, bytearray)) or not isinstance(
        object_rows, Sequence
    ):
        raise ResearchArgumentError("Research Object rows must be a sequence")
    research_nodes: dict[str, Mapping[str, Any]] = {}
    for node in graph.get("nodes") or []:
        if not isinstance(node, Mapping) or node.get("source") != "research_vcs":
            continue
        object_id = str(node.get("id") or "")
        if object_id in research_nodes:
            raise ResearchArgumentError(
                f"duplicate authoritative Research VCS node: {object_id}"
            )
        research_nodes[object_id] = node

    objects: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(object_rows):
        if not isinstance(item, Mapping):
            raise ResearchArgumentError(f"Research Object row[{index}] is invalid")
        try:
            validated = validate_research_object(item)
        except (ResearchObjectError, TypeError, ValueError) as exc:
            object_id = str(item.get("object_id") or f"row[{index}]")
            raise ResearchArgumentError(
                f"Research Object authoritative identity is invalid: {object_id}"
            ) from exc
        object_id = str(item.get("object_id") or "")
        if not object_id or object_id in objects:
            raise ResearchArgumentError(
                f"Research Object row identity is missing or duplicated: {object_id}"
            )
        if not _content_hash(item.get("content_hash")):
            raise ResearchArgumentError(
                f"Research Object content_hash is invalid: {object_id}"
            )
        objects[object_id] = validated

    if set(objects) != set(research_nodes):
        raise ResearchArgumentError(
            "Research Object rows do not exactly match authoritative DAG nodes"
        )
    commit = str(graph.get("commit") or "")
    for object_id, item in objects.items():
        node = research_nodes[object_id]
        if (
            node.get("kind") != item.get("kind")
            or node.get("state") != item.get("state")
            or node.get("content_hash") != item.get("content_hash")
            or node.get("source_ref") != commit
        ):
            raise ResearchArgumentError(
                f"Research Object identity mismatch for authoritative node: {object_id}"
            )
    return objects


def _validated_closure(
    graph: Mapping[str, Any],
    closure: Mapping[str, Any] | None,
    *,
    objects: Mapping[str, Mapping[str, Any]] | None = None,
) -> Mapping[str, Any] | None:
    """Validate the payload-free closure authority committed by the DAG."""

    graph_claim_ids = {
        str(node.get("id") or "")
        for node in graph.get("nodes") or []
        if isinstance(node, Mapping)
        and node.get("source") == "research_vcs"
        and node.get("kind") == "claim"
    }
    graph_summary = graph.get("scientific_closure")
    if not isinstance(graph_summary, Mapping):
        raise ResearchArgumentError("research DAG scientific_closure is invalid")
    if not graph_claim_ids:
        if closure is not None:
            raise ResearchArgumentError(
                "research closure is present without authoritative claim nodes"
            )
        if graph_summary.get("content_hash") not in {None, ""}:
            raise ResearchArgumentError(
                "research DAG commits a closure without authoritative claim nodes"
            )
        return None
    if not isinstance(closure, Mapping):
        raise ResearchArgumentError(
            "registered claims require the DAG-committed research closure"
        )
    normalized = dict(closure)
    try:
        validate_json(normalized, load_schema("research_closure"))
    except ValidationError as exc:
        raise ResearchArgumentError(
            f"research closure schema is invalid: {exc.message}"
        ) from exc
    if normalized.get("schema_version") != RESEARCH_CLOSURE_SCHEMA:
        raise ResearchArgumentError("unsupported research closure schema_version")
    if normalized.get("commit") != graph.get("commit"):
        raise ResearchArgumentError("research closure commit does not match DAG commit")
    if normalized.get("target_level") != "verify":
        raise ResearchArgumentError("research closure must bind the verify level")
    closure_core = {
        key: value for key, value in normalized.items() if key != "content_hash"
    }
    try:
        expected_hash = protocol_content_hash(closure_core)
    except (TypeError, ValueError) as exc:
        raise ResearchArgumentError(
            "research closure hash scope is not canonical JSON"
        ) from exc
    if normalized.get("content_hash") != expected_hash:
        raise ResearchArgumentError("research closure content_hash mismatch")
    if graph_summary.get("content_hash") != expected_hash:
        raise ResearchArgumentError("research closure does not match DAG commitment")
    expected_summary = {
        "status": normalized.get("status"),
        "blocker_count": len(normalized.get("blockers") or []),
        "warning_count": len(normalized.get("warnings") or []),
        "closure_levels": dict(normalized.get("closure_levels") or {}),
        "content_hash": expected_hash,
    }
    for key, expected in expected_summary.items():
        if graph_summary.get(key) != expected:
            raise ResearchArgumentError(
                f"research closure {key} does not match DAG projection"
            )

    rows = normalized.get("claims") or []
    closure_claim_ids = [str(row.get("claim_id") or "") for row in rows]
    if any(not claim_id for claim_id in closure_claim_ids) or len(
        closure_claim_ids
    ) != len(set(closure_claim_ids)):
        raise ResearchArgumentError("research closure claim identities are invalid")
    if objects is not None:
        all_claims = [
            item for item in objects.values() if item.get("kind") == "claim"
        ]
        frontier, superseded_claim_ids = _effective_claim_frontier(
            all_claims, objects
        )
        frontier_ids = sorted(str(item["object_id"]) for item in frontier)
        if sorted(closure_claim_ids) != frontier_ids:
            raise ResearchArgumentError(
                "research closure claims do not match the active claim frontier"
            )
        if sorted(normalized.get("superseded_claim_ids") or []) != sorted(
            superseded_claim_ids
        ):
            raise ResearchArgumentError(
                "research closure superseded claims do not match Research VCS"
            )
        for row in rows:
            claim_id = str(row["claim_id"])
            claim = objects.get(claim_id)
            if claim is None or row.get("state") != claim.get("state"):
                raise ResearchArgumentError(
                    f"research closure claim state is stale: {claim_id}"
                )
    elif not set(closure_claim_ids).issubset(graph_claim_ids):
        raise ResearchArgumentError(
            "research closure references a claim absent from the DAG"
        )
    return normalized


def _research_binding(
    graph: Mapping[str, Any], nodes: Sequence[Any], edges: Sequence[Any]
) -> dict[str, Any]:
    return {
        "schema_version": str(graph.get("schema_version") or ""),
        "ref": str(graph.get("ref") or ""),
        "commit": str(graph.get("commit") or ""),
        "graph_hash": str(graph.get("graph_hash") or ""),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "content_disclosed": graph.get("content_disclosed") is True,
    }


def validate_research_argument_view(
    graph: Mapping[str, Any], argument_view: Mapping[str, Any]
) -> None:
    """Reject a sidecar unless all authoritative projections are reproducible."""

    if not isinstance(graph, Mapping) or not isinstance(argument_view, Mapping):
        raise ResearchArgumentError("research DAG and argument sidecar must be objects")
    nodes, edges = _validated_graph_collections(graph)
    if argument_view.get("schema_version") != RESEARCH_ARGUMENT_SCHEMA:
        raise ResearchArgumentError("unsupported research argument schema_version")
    declared_hash = argument_view.get("content_hash")
    if not _content_hash(declared_hash):
        raise ResearchArgumentError("research argument content_hash is invalid")
    sidecar_core = {
        key: value for key, value in argument_view.items() if key != "content_hash"
    }
    try:
        expected_hash = canonical_content_hash(sidecar_core)
    except (TypeError, ValueError) as exc:
        raise ResearchArgumentError(
            "research argument hash scope is not canonical JSON"
        ) from exc
    if declared_hash != expected_hash:
        raise ResearchArgumentError("research argument content_hash mismatch")
    binding = argument_view.get("research_binding")
    expected_binding = _research_binding(graph, nodes, edges)
    if not isinstance(binding, Mapping) or dict(binding) != expected_binding:
        raise ResearchArgumentError("research argument DAG binding mismatch")

    summaries_disclosed = argument_view.get("summaries_disclosed")
    if not isinstance(summaries_disclosed, bool):
        # Legacy claim-free views contain no payload-derived claim metadata and
        # remain safe to render.  Claim-bearing views require an explicit mode.
        if argument_view.get("registered_claims"):
            raise ResearchArgumentError(
                "research argument summaries_disclosed is invalid"
            )
        summaries_disclosed = graph.get("content_disclosed") is True
    if summaries_disclosed and graph.get("content_disclosed") is not True:
        raise ResearchArgumentError(
            "research argument cannot disclose summaries hidden by the DAG"
        )
    closure = _validated_closure(
        graph, argument_view.get("scientific_closure_authority")
    )
    expected_claims = _canonical_registered_claims(
        graph, closure, disclose_summaries=summaries_disclosed
    )
    if argument_view.get("registered_claims") != expected_claims:
        raise ResearchArgumentError(
            "research argument registered claims are not the canonical DAG projection"
        )
    advisory_routes = argument_view.get("advisory_routes")
    if not isinstance(advisory_routes, list) or any(
        not isinstance(route, Mapping) for route in advisory_routes
    ):
        raise ResearchArgumentError("research argument advisory routes are invalid")
    expected_counters = _canonical_counters(
        graph,
        nodes,
        edges,
        registered_claims=expected_claims,
        advisory_routes=advisory_routes,
    )
    if argument_view.get("counters") != expected_counters:
        raise ResearchArgumentError(
            "research argument counters are not the canonical DAG projection"
        )


def _relation_targets(item: Mapping[str, Any], relation_type: str) -> set[str]:
    return {
        str(relation.get("target") or "")
        for relation in item.get("relations") or []
        if isinstance(relation, Mapping) and relation.get("type") == relation_type
    } - {""}


def _reproduction_bindings(
    claim_id: str,
    reproduction_ids: Sequence[str],
    objects: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reproduction_id in sorted(set(reproduction_ids)):
        reproduction = objects.get(reproduction_id) or {}
        payload = reproduction.get("payload") or {}
        receipt = payload.get("receipt") or {}
        target_binding = receipt.get("target_binding") or {}
        claim_bindings = [
            dict(item)
            for item in target_binding.get("claim_closures") or []
            if isinstance(item, Mapping) and item.get("claim_id") == claim_id
        ]
        rows.append(
            {
                "reproduction_id": reproduction_id,
                "state": str(reproduction.get("state") or ""),
                "receipt_hash": str(payload.get("receipt_hash") or "") or None,
                "claim_closure_bindings": claim_bindings,
            }
        )
    return rows


def _source_for_ara_binding(
    graph: Mapping[str, Any], manifest_hash: str, graph_hash: str
) -> Mapping[str, Any] | None:
    matching = []
    for source in graph.get("sources") or []:
        if not isinstance(source, Mapping) or not re.fullmatch(
            r"ara:\d+", str(source.get("name") or "")
        ):
            continue
        manifest_matches = source.get("manifest_hash") == manifest_hash
        graph_binding = source.get("graph_binding") or {}
        manifest_integrity = source.get("manifest_integrity") or {}
        integrity = source.get("integrity") or {}
        if (
            manifest_matches
            and graph_binding.get("graph_hash") == graph_hash
            and manifest_integrity.get("ok") is True
            and manifest_integrity.get("state") == "clean"
            and graph_binding.get("ok") is True
            and graph_binding.get("state") in {"verified", "commit_bound"}
            and integrity.get("error_count", 0) == 0
        ):
            matching.append(source)
    return matching[0] if len(matching) == 1 else None


def _object_has_ara_pair(
    item: Mapping[str, Any], manifest_hash: str, graph_hash: str
) -> bool:
    for container_name in ("payload", "provenance"):
        container = item.get(container_name) or {}
        if (
            isinstance(container, Mapping)
            and container.get("ara_manifest_hash") == manifest_hash
            and container.get("ara_exploration_graph_hash") == graph_hash
        ):
            return True
    return False


def _role_binding_shape(value: Any) -> tuple[Mapping[str, Any] | None, list[str]]:
    if not isinstance(value, Mapping):
        return None, ["binding_not_object"]
    common = {
        "schema_version",
        "claim_id",
        "manifest_hash",
        "exploration_graph_hash",
        "selected_node_ids",
        "selected_node_ids_hash",
        "role",
        "metric",
        "control_ids",
        "execution_receipt_hash",
    }
    keys = set(value)
    if keys not in (common | {"reproduction_id"}, common | {"reproduction_state"}):
        return None, ["binding_keys_invalid"]
    issues: list[str] = []
    if value.get("schema_version") != ARA_CLAIM_ROLE_BINDING_SCHEMA:
        issues.append("binding_schema_invalid")
    for key in ("claim_id", "manifest_hash", "exploration_graph_hash"):
        if not isinstance(value.get(key), str):
            issues.append(f"{key}_invalid")
    if not _content_hash(value.get("manifest_hash")):
        issues.append("manifest_hash_invalid")
    if not _content_hash(value.get("exploration_graph_hash")):
        issues.append("exploration_graph_hash_invalid")
    if not _content_hash(value.get("selected_node_ids_hash")):
        issues.append("selected_node_ids_hash_invalid")
    if not _content_hash(value.get("execution_receipt_hash")):
        issues.append("execution_receipt_hash_invalid")
    selected = value.get("selected_node_ids")
    if (
        not isinstance(selected, list)
        or not selected
        or len(selected) > _MAX_ADVISORY_FILES
        or any(
            not isinstance(item, str) or not item or len(item) > _MAX_SHORT_TEXT
            for item in selected
        )
        or len(set(selected)) != len(selected)
    ):
        issues.append("selected_node_ids_invalid")
    else:
        try:
            selected_hash_matches = value.get(
                "selected_node_ids_hash"
            ) == canonical_content_hash(selected)
        except (TypeError, ValueError):
            selected_hash_matches = False
        if not selected_hash_matches:
            issues.append("selected_node_ids_hash_mismatch")
    if value.get("role") not in _ROLE_EDGE_CATEGORY:
        issues.append("role_invalid")
    metric = value.get("metric")
    if not isinstance(metric, Mapping) or set(metric) != {
        "name",
        "direction",
        "threshold",
    }:
        issues.append("metric_invalid")
    else:
        if (
            not isinstance(metric.get("name"), str)
            or not metric.get("name")
            or len(metric["name"]) > _MAX_SHORT_TEXT
        ):
            issues.append("metric_name_invalid")
        if metric.get("direction") not in {"maximize", "minimize", "equal"}:
            issues.append("metric_direction_invalid")
        threshold = metric.get("threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
        ):
            issues.append("metric_threshold_invalid")
    controls = value.get("control_ids")
    if (
        not isinstance(controls, list)
        or len(controls) > _MAX_ADVISORY_FILES
        or any(
            not isinstance(item, str) or not item or len(item) > _MAX_SHORT_TEXT
            for item in controls
        )
        or len(set(controls)) != len(controls)
    ):
        issues.append("control_ids_invalid")
    if "reproduction_id" in value:
        if not isinstance(value["reproduction_id"], str) or not _RSO_ID_RE.fullmatch(
            value["reproduction_id"]
        ):
            issues.append("reproduction_id_invalid")
    elif value.get("reproduction_state") != "NOT_REPRODUCED":
        issues.append("reproduction_state_invalid")
    return (value if not issues else None), sorted(set(issues))


def _semantic_edge_present(
    edges: Sequence[Mapping[str, Any]],
    *,
    claim_id: str,
    evidence_id: str,
    category: str,
    relation_types: set[str],
) -> bool:
    return any(
        edge.get("source") == claim_id
        and edge.get("target") == evidence_id
        and edge.get("category") == category
        and edge.get("type") in relation_types
        for edge in edges
    )


def _object_semantic_relation_present(
    item: Mapping[str, Any], claim_id: str, relation_types: set[str]
) -> bool:
    return any(
        isinstance(relation, Mapping)
        and relation.get("target") == claim_id
        and relation.get("type") in relation_types
        for relation in item.get("relations") or []
    )


def _ara_node_proof_is_exact(node: Mapping[str, Any]) -> bool:
    proof = node.get("proof")
    if not isinstance(proof, Mapping):
        return False
    checks = proof.get("checks")
    return bool(
        proof.get("level") in {"replayable", "verified"}
        and proof.get("contested") is False
        and not (proof.get("blockers") or [])
        and isinstance(checks, list)
        and checks
        and all(
            isinstance(check, Mapping) and check.get("passed") is True
            for check in checks
        )
    )


def _ara_role_bindings(
    *,
    graph: Mapping[str, Any],
    claim_id: str,
    closure_evidence_ids: set[str],
    reproduction_ids: set[str],
    objects: Mapping[str, Mapping[str, Any]],
    disclose_summaries: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    graph_nodes = {
        str(node.get("id") or ""): node
        for node in graph.get("nodes") or []
        if isinstance(node, Mapping)
        and node.get("source") == "ara"
        and node.get("kind") == "experiment_node"
    }
    graph_edges = [
        edge for edge in graph.get("edges") or [] if isinstance(edge, Mapping)
    ]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    bound_nodes: set[str] = set()
    for evidence_id in sorted(closure_evidence_ids):
        evidence = objects.get(evidence_id) or {}
        raw_bindings = (evidence.get("payload") or {}).get("ara_claim_role_bindings")
        if raw_bindings is None:
            continue
        if not isinstance(raw_bindings, list):
            rejected.append(
                {
                    "evidence_object_id": evidence_id,
                    "reason_codes": ["bindings_not_list"],
                }
            )
            continue
        for binding_number, raw_binding in enumerate(raw_bindings):
            binding, issues = _role_binding_shape(raw_binding)
            if binding is None:
                rejected.append(
                    {
                        "evidence_object_id": evidence_id,
                        "binding_index": binding_number,
                        "reason_codes": issues,
                    }
                )
                continue
            manifest_hash = str(binding["manifest_hash"])
            exploration_graph_hash = str(binding["exploration_graph_hash"])
            source = _source_for_ara_binding(
                graph, manifest_hash, exploration_graph_hash
            )
            expected_category = _ROLE_EDGE_CATEGORY[str(binding["role"])]
            expected_relation_types = _ROLE_EDGE_TYPES[str(binding["role"])]
            # The v1 unified DAG does not expose a typed, immutable authority
            # record for node metrics, control semantics, or execution receipts.
            # Shape-valid declarations therefore remain visible only as rejected
            # metadata and can never increase the exact-role numerator.
            reason_codes: list[str] = [
                "metric_authority_unavailable",
                "control_ids_unresolved",
                "execution_receipt_unresolved",
            ]
            if evidence.get("kind") not in {"evidence", "passage_evidence"}:
                reason_codes.append("object_not_evidence")
            if evidence.get("state") not in _ACTIVE_EVIDENCE_STATES:
                reason_codes.append("evidence_inactive")
            if evidence_id not in closure_evidence_ids:
                reason_codes.append("evidence_outside_claim_closure")
            if binding.get("claim_id") != claim_id:
                reason_codes.append("claim_id_mismatch")
            if not _object_has_ara_pair(
                evidence, manifest_hash, exploration_graph_hash
            ):
                reason_codes.append("evidence_ara_pair_unbound")
            if source is None:
                reason_codes.append("ara_source_not_clean_and_exact")
                projected_ids: list[str] = []
            else:
                projected_ids = [
                    f"{source['name']}:{raw_id}"
                    for raw_id in binding["selected_node_ids"]
                ]
                selected_nodes = [graph_nodes.get(node_id) for node_id in projected_ids]
                if any(node is None for node in selected_nodes):
                    reason_codes.append("selected_ara_node_unresolved")
                else:
                    for node in selected_nodes:
                        assert node is not None
                        if node.get("state") != "completed":
                            reason_codes.append("selected_ara_node_state_invalid")
                        if not _content_hash(node.get("content_hash")):
                            reason_codes.append(
                                "selected_ara_node_content_hash_invalid"
                            )
                        if not _ara_node_proof_is_exact(node):
                            reason_codes.append("selected_ara_node_proof_invalid")
                projected_control_ids = [
                    f"{source['name']}:{raw_id}" for raw_id in binding["control_ids"]
                ]
                if any(
                    control_id not in graph_nodes
                    for control_id in projected_control_ids
                ):
                    reason_codes.append("control_ara_node_unresolved")
            if not _semantic_edge_present(
                graph_edges,
                claim_id=claim_id,
                evidence_id=evidence_id,
                category=expected_category,
                relation_types=expected_relation_types,
            ) or not _object_semantic_relation_present(
                evidence, claim_id, expected_relation_types
            ):
                reason_codes.append("semantic_relation_missing")
            reproduction_id = binding.get("reproduction_id")
            if reproduction_id is not None and reproduction_id not in reproduction_ids:
                reason_codes.append("reproduction_outside_claim_closure")
            if reason_codes:
                rejected.append(
                    {
                        "evidence_object_id": evidence_id,
                        "binding_index": binding_number,
                        "reason_codes": sorted(set(reason_codes)),
                    }
                )
                continue
            projected = {
                **dict(binding),
                "evidence_object_id": evidence_id,
                "source_name": str(source["name"]),
                "projected_node_ids": projected_ids,
            }
            if not disclose_summaries:
                projected.pop("metric", None)
            accepted.append(projected)
            bound_nodes.update(projected_ids)
    accepted.sort(
        key=lambda item: (
            str(item.get("claim_id") or ""),
            str(item.get("evidence_object_id") or ""),
            str(item.get("role") or ""),
            tuple(item.get("projected_node_ids") or []),
        )
    )
    rejected.sort(
        key=lambda item: (
            str(item.get("evidence_object_id") or ""),
            int(item.get("binding_index") or 0),
        )
    )
    return accepted, rejected, bound_nodes


def _graph_node_index(graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(node["id"]): node
        for node in graph.get("nodes") or []
        if isinstance(node, Mapping) and node.get("id")
    }


def _graph_claim_alias_ids(
    claim_id: str,
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> set[str]:
    """Recover explicit immutable claim aliases from supersession edges."""

    aliases = {claim_id}
    pending = [claim_id]
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if (
            edge.get("type") != "supersedes"
            or edge.get("category") != "evolution"
            or nodes.get(source, {}).get("kind") != "claim"
            or nodes.get(target, {}).get("kind") != "claim"
        ):
            continue
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    while pending:
        current = pending.pop()
        for neighbor in sorted(adjacency.get(current, set()) - aliases):
            aliases.add(neighbor)
            pending.append(neighbor)
    return aliases


def _graph_superseded_ids(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> set[str]:
    superseded = {
        object_id
        for object_id, node in nodes.items()
        if node.get("source") == "research_vcs" and node.get("state") == "superseded"
    }
    superseded.update(
        str(edge.get("source") or "")
        for edge in edges
        if edge.get("type") == "supersedes"
        and edge.get("category") == "evolution"
        and nodes.get(str(edge.get("target") or ""), {}).get("state")
        in _ACTIVE_EVIDENCE_STATES
    )
    return superseded - {""}


def _graph_direct_semantic_ids(
    *,
    claim_id: str,
    closure_member_ids: set[str],
    category: str,
    relation_types: set[str],
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    superseded_ids: set[str],
) -> list[str]:
    aliases = _graph_claim_alias_ids(claim_id, nodes, edges)
    return sorted(
        {
            str(edge.get("target") or "")
            for edge in edges
            if str(edge.get("source") or "") in aliases
            and str(edge.get("target") or "") in closure_member_ids
            and edge.get("category") == category
            and edge.get("type") in relation_types
            and str(edge.get("target") or "") not in superseded_ids
            and nodes.get(str(edge.get("target") or ""), {}).get("state")
            in _ACTIVE_EVIDENCE_STATES
        }
        - {""}
    )


def _object_direct_semantic_ids(
    *,
    claim: Mapping[str, Any],
    closure_member_ids: set[str],
    relation_types: set[str],
    objects: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    aliases = _claim_alias_ids(claim, objects)
    superseded_ids = _effective_superseded_ids(objects)
    return sorted(
        object_id
        for object_id in closure_member_ids
        if _is_active_closure_object(object_id, objects, superseded_ids)
        and any(
            isinstance(relation, Mapping)
            and relation.get("type") in relation_types
            and str(relation.get("target") or "") in aliases
            for relation in objects.get(object_id, {}).get("relations") or []
        )
    )


def _canonical_registered_claims(
    graph: Mapping[str, Any],
    closure: Mapping[str, Any] | None,
    *,
    disclose_summaries: bool,
) -> list[dict[str, Any]]:
    if closure is None:
        return []
    nodes = _graph_node_index(graph)
    edges = [
        edge for edge in graph.get("edges") or [] if isinstance(edge, Mapping)
    ]
    superseded_ids = _graph_superseded_ids(nodes, edges)
    insight_by_claim: dict[str, Mapping[str, Any]] = {}
    for insight in graph.get("claim_insights") or []:
        if not isinstance(insight, Mapping) or not insight.get("claim_id"):
            raise ResearchArgumentError("research DAG claim insight is invalid")
        claim_id = str(insight["claim_id"])
        if claim_id in insight_by_claim:
            raise ResearchArgumentError("research DAG claim insights are duplicated")
        insight_by_claim[claim_id] = insight

    registered: list[dict[str, Any]] = []
    for row in closure.get("claims") or []:
        claim_id = str(row["claim_id"])
        claim_node = nodes.get(claim_id)
        if (
            claim_node is None
            or claim_node.get("source") != "research_vcs"
            or claim_node.get("kind") != "claim"
        ):
            raise ResearchArgumentError(
                f"research closure claim is absent from DAG: {claim_id}"
            )
        closure_evidence_ids = set(row.get("evidence_ids") or [])
        closure_evidence_ids.update(row.get("passage_evidence_ids") or [])
        closure_argument_ids = set(row.get("argument_ids") or [])
        closure_member_ids = {*closure_evidence_ids, *closure_argument_ids}
        direct_support = _graph_direct_semantic_ids(
            claim_id=claim_id,
            closure_member_ids=closure_member_ids,
            category="support",
            relation_types=_SUPPORT_RELATION_TYPES,
            nodes=nodes,
            edges=edges,
            superseded_ids=superseded_ids,
        )
        direct_challenge = _graph_direct_semantic_ids(
            claim_id=claim_id,
            closure_member_ids=closure_member_ids,
            category="challenge",
            relation_types=_CHALLENGE_RELATION_TYPES,
            nodes=nodes,
            edges=edges,
            superseded_ids=superseded_ids,
        )
        argument_kinds = {
            kind: sorted(
                object_id
                for object_id in closure_argument_ids
                if nodes.get(object_id, {}).get("kind") == kind
            )
            for kind in (
                "assumption",
                "boundary_condition",
                "estimand",
                "inference",
                "transfer_matrix",
                "warrant",
            )
        }
        gate_ids = sorted(set(row.get("gate_ids") or []))
        review_ids = sorted(
            {
                str(edge.get("source") or "")
                for edge in edges
                if str(edge.get("target") or "") in gate_ids
                and edge.get("type") == "evaluates"
                and edge.get("category") == "verification"
                and nodes.get(str(edge.get("source") or ""), {}).get("kind")
                == "review"
                and (
                    nodes.get(str(edge.get("source") or ""), {}).get("actor") or {}
                ).get("authority")
                == "independent_evaluator"
            }
            - {""}
        )
        supporting_evidence = [
            object_id
            for object_id in direct_support
            if nodes.get(object_id, {}).get("kind")
            in {"evidence", "passage_evidence"}
        ]
        challenging_evidence = [
            object_id
            for object_id in direct_challenge
            if nodes.get(object_id, {}).get("kind")
            in {"evidence", "passage_evidence"}
        ]
        gaps = set(row.get("missing") or [])
        if not direct_support:
            gaps.add("direct_semantic_support_missing")
        if direct_challenge:
            gaps.add("direct_semantic_challenge_active")
        insight = insight_by_claim.get(claim_id) or {}
        card: dict[str, Any] = {
            "claim_id": claim_id,
            "state": str(claim_node.get("state") or ""),
            "depth_level": str(insight.get("depth_level") or "descriptive"),
            # The current DAG commits only a claim summary, not a Merkle proof for
            # arbitrary payload fields.  Do not project unauthenticated scope data.
            "scope_hash": None,
            "closure_state": "active",
            "closure_evidence_ids": sorted(closure_evidence_ids),
            "direct_supporting_ids": direct_support,
            "direct_challenging_ids": direct_challenge,
            "supporting_evidence_ids": supporting_evidence,
            "supporting_argument_ids": sorted(
                set(direct_support) - set(supporting_evidence)
            ),
            "challenging_evidence_ids": challenging_evidence,
            "challenging_argument_ids": sorted(
                set(direct_challenge) - set(challenging_evidence)
            ),
            "inference_ids": argument_kinds["inference"],
            "warrant_ids": argument_kinds["warrant"],
            "assumption_ids": argument_kinds["assumption"],
            "estimand_ids": argument_kinds["estimand"],
            "boundary_ids": sorted(
                {
                    *argument_kinds["boundary_condition"],
                    *argument_kinds["transfer_matrix"],
                }
            ),
            "independent_review_ids": review_ids,
            "gate_ids": gate_ids,
            "reproductions": [
                {
                    "reproduction_id": reproduction_id,
                    "state": str(nodes.get(reproduction_id, {}).get("state") or ""),
                    "receipt_hash": None,
                    "claim_closure_bindings": [],
                }
                for reproduction_id in sorted(set(row.get("reproduction_ids") or []))
            ],
            # ARA v1 has no typed metric/control/receipt authority.  The only
            # renderer-verifiable accepted set is therefore the empty set.
            "ara_claim_role_bindings": [],
            "ara_binding_rejections": [],
            "gaps": sorted(gaps),
            "decision_ready": bool(
                row.get("verified") is True
                and direct_support
                and not direct_challenge
                and not gaps
            ),
        }
        if disclose_summaries:
            card["statement"] = str(claim_node.get("summary") or "")
        registered.append(card)
    return sorted(registered, key=lambda item: str(item["claim_id"]))


def _registered_claims(
    graph: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
    closure: Mapping[str, Any] | None,
    *,
    disclose_summaries: bool,
) -> tuple[list[dict[str, Any]], set[str]]:
    registered = _canonical_registered_claims(
        graph, closure, disclose_summaries=disclose_summaries
    )
    closure_rows = {
        str(item["claim_id"]): item for item in (closure or {}).get("claims") or []
    }
    for card in registered:
        claim_id = str(card["claim_id"])
        row = closure_rows[claim_id]
        claim = objects.get(claim_id)
        if claim is None:
            raise ResearchArgumentError(
                f"active closure claim is absent from Research Objects: {claim_id}"
            )
        closure_members = {
            *(row.get("evidence_ids") or []),
            *(row.get("passage_evidence_ids") or []),
            *(row.get("argument_ids") or []),
        }
        object_support = _object_direct_semantic_ids(
            claim=claim,
            closure_member_ids=closure_members,
            relation_types=_SUPPORT_RELATION_TYPES,
            objects=objects,
        )
        object_challenge = _object_direct_semantic_ids(
            claim=claim,
            closure_member_ids=closure_members,
            relation_types=_CHALLENGE_RELATION_TYPES,
            objects=objects,
        )
        if object_support != card["direct_supporting_ids"]:
            raise ResearchArgumentError(
                f"DAG support edges disagree with active claim closure: {claim_id}"
            )
        if object_challenge != card["direct_challenging_ids"]:
            raise ResearchArgumentError(
                f"DAG challenge edges disagree with active claim closure: {claim_id}"
            )
    return registered, set()


def _canonical_counters(
    graph: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    *,
    registered_claims: Sequence[Mapping[str, Any]],
    advisory_routes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    categories: dict[str, int] = {}
    for edge in edges:
        category = str(edge.get("category") or "")
        categories[category] = categories.get(category, 0) + 1
    categories = dict(sorted(categories.items()))
    total_edges = len(edges)
    ara_total = sum(
        1
        for node in nodes
        if node.get("source") == "ara" and node.get("kind") == "experiment_node"
    )
    pool = (graph.get("theory_frontier") or {}).get("submission_candidate_pool") or {}
    candidate_count = pool.get("candidate_count")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        candidate_count = None
    experiment_authorized = pool.get("experiment_authorized")
    if not isinstance(experiment_authorized, bool):
        experiment_authorized = None
    return {
        "registered_claims": len(registered_claims),
        "unregistered_routes": len(advisory_routes),
        "closure": str(
            (graph.get("scientific_closure") or {}).get("status")
            or "not_applicable"
        ),
        "submission_candidates": candidate_count,
        "experiment_authorized": experiment_authorized,
        "direct_semantic_edges": {
            "count": sum(
                categories.get(category, 0)
                for category in _DIRECT_SEMANTIC_CATEGORIES
            ),
            "total": total_edges,
        },
        "verification_edges": {
            "count": categories.get("verification", 0),
            "total": total_edges,
        },
        "provenance_context_evolution_edges": {
            "count": sum(
                categories.get(category, 0) for category in _PROVENANCE_CATEGORIES
            ),
            "total": total_edges,
        },
        "ara_claim_role_nodes": {"bound": 0, "total": ara_total},
        "edge_categories": categories,
    }


def build_research_argument_view(
    graph: Mapping[str, Any],
    *,
    object_rows: Sequence[Mapping[str, Any]] = (),
    closure: Mapping[str, Any] | None = None,
    repo: str | Path | None = None,
    advisory_index: str | Path | None = None,
    disclose_summaries: bool | None = None,
) -> dict[str, Any]:
    """Build a deterministic sidecar without mutating the authoritative graph."""

    if not isinstance(graph, Mapping):
        raise ResearchArgumentError("research DAG must be an object")
    nodes, edges = _validated_graph_collections(graph)
    objects = _validated_research_objects(graph, object_rows)
    validated_closure = _validated_closure(graph, closure, objects=objects)
    requested_disclosure = (
        True if disclose_summaries is None else bool(disclose_summaries)
    )
    effective_disclosure = bool(
        requested_disclosure and graph.get("content_disclosed") is True
    )

    registered_claims, _bound_ara_nodes = _registered_claims(
        graph,
        objects,
        validated_closure,
        disclose_summaries=effective_disclosure,
    )
    advisory_projection: dict[str, Any] | None = None
    if advisory_index is not None:
        if repo is None:
            raise ResearchArgumentError(
                "repo is required when advisory_index is supplied"
            )
        advisory_projection = project_advisory_route_index(
            repo,
            advisory_index,
            disclose_summaries=effective_disclosure,
        )
    advisory_routes = (advisory_projection or {}).get("routes") or []
    base = {
        "schema_version": RESEARCH_ARGUMENT_SCHEMA,
        "research_binding": _research_binding(graph, nodes, edges),
        "scientific_closure_authority": (
            dict(validated_closure) if validated_closure is not None else None
        ),
        "summaries_disclosed": effective_disclosure,
        "advisory_index_binding": ((advisory_projection or {}).get("index_binding")),
        "registered_claims": registered_claims,
        "advisory_routes": advisory_routes,
        "counters": _canonical_counters(
            graph,
            nodes,
            edges,
            registered_claims=registered_claims,
            advisory_routes=advisory_routes,
        ),
    }
    return {**base, "content_hash": canonical_content_hash(base)}


# A concise alias for callers that treat every non-authoritative view as a
# projection.  Both names retain the same strict contract.
project_research_argument = build_research_argument_view


__all__ = [
    "ADVISORY_ROUTE_INDEX_SCHEMA",
    "ARA_CLAIM_ROLE_BINDING_SCHEMA",
    "RESEARCH_ARGUMENT_SCHEMA",
    "ResearchArgumentError",
    "build_research_argument_view",
    "load_advisory_route_index",
    "project_advisory_route_index",
    "project_research_argument",
    "validate_research_argument_view",
]
