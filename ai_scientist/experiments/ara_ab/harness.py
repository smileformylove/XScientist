"""A/B harness — does an ARA seed actually accelerate the next agent?

Two arms, one idea, one metric board:

    baseline   → cold-start XScientist run, no ARA seed.
    ara_seed   → seeded via --seed-from-ara from a parent ARA node.

We compare wall-clock, LLM invocations, node count, buggy ratio, and the
`content_hash` overlap between the two arms' final graphs. Results land in
`<work_dir>/ab_report.json` and get echoed to stdout.

Modes
-----
- ``stub``  Ships with the repo; no API keys or GPUs required. Instantiates
            ``MinimalAgent._draft`` twice against a fake plan-and-code query
            and asserts the seed short-circuits. Runs in CI.
- ``real``  Shells out to ``run_project.py`` twice with matching flags. You
            pay for the LLM calls. Not run in CI.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import types
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ArmResult:
    """One side of the A/B run."""

    arm: str
    started_at: str
    finished_at: str
    duration_seconds: float
    llm_calls: int
    used_seed: bool
    exit_code: int = 0
    ara_root: str | None = None
    node_count: int | None = None
    buggy_ratio: float | None = None
    node_content_hashes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ABReport:
    schema: str = "ara.ab_report.v1"
    generated_at: str = field(default_factory=_now_iso)
    idea_hint: str | None = None
    mode: str = "stub"
    baseline: ArmResult | None = None
    ara_seed: ArmResult | None = None
    hash_overlap: dict[str, Any] = field(default_factory=dict)
    verdict: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "generated_at": self.generated_at,
            "idea_hint": self.idea_hint,
            "mode": self.mode,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "ara_seed": self.ara_seed.to_dict() if self.ara_seed else None,
            "hash_overlap": self.hash_overlap,
            "verdict": self.verdict,
        }


# ---------------------------------------------------------------------------
# Stub mode — CI-safe, exercises the seed short-circuit without a real LLM.
# ---------------------------------------------------------------------------


def _make_stub_agent(*, seed_code: str | None):
    """Build a minimal duck-typed object with `_draft` bound from MinimalAgent.

    This mirrors what tests/test_ara_seed.py::DraftShortCircuitTest does, but
    packaged so callers can measure both arms uniformly.
    """
    from ai_scientist.treesearch import parallel_agent

    counter = {"llm_calls": 0}

    def _fake_plan_and_code(self, prompt):
        counter["llm_calls"] += 1
        code = seed_code or "print('baseline draft')"
        return "baseline plan", code

    class _StubMinimalAgent:
        _draft = parallel_agent.MinimalAgent._draft
        plan_and_code_query = _fake_plan_and_code
        cfg = types.SimpleNamespace(agent=types.SimpleNamespace(data_preview=False))
        task_desc = "ab harness stub task"
        memory_summary = ""
        _prompt_resp_fmt: dict[str, Any] = {}
        _prompt_impl_guideline: dict[str, Any] = {}
        _prompt_environment: dict[str, Any] = {}
        evaluation_metrics = "acc"
        data_preview = ""

    return _StubMinimalAgent, counter


def run_stub_arm(
    *,
    arm: str,
    seed_manifest_path: Path | None,
) -> ArmResult:
    """Execute the seed short-circuit path once and record what happened."""
    import os

    from ai_scientist.utils.ara_seed import SEED_ENV_VAR

    orig_env = os.environ.get(SEED_ENV_VAR)
    try:
        if seed_manifest_path is not None:
            os.environ[SEED_ENV_VAR] = str(seed_manifest_path)
        else:
            os.environ.pop(SEED_ENV_VAR, None)

        started_at = _now_iso()
        started_perf = time.perf_counter()
        Agent, counter = _make_stub_agent(seed_code=None)
        node = Agent._draft(Agent())  # type: ignore[arg-type]
        finished_perf = time.perf_counter()

        seed_hit = counter["llm_calls"] == 0 and seed_manifest_path is not None
        return ArmResult(
            arm=arm,
            started_at=started_at,
            finished_at=_now_iso(),
            duration_seconds=finished_perf - started_perf,
            llm_calls=counter["llm_calls"],
            used_seed=seed_hit,
            node_count=1,
            buggy_ratio=0.0,
            node_content_hashes=[],
            notes=[
                f"stub draft code prefix: {node.code[:40]!r}",
            ],
        )
    finally:
        if orig_env is None:
            os.environ.pop(SEED_ENV_VAR, None)
        else:
            os.environ[SEED_ENV_VAR] = orig_env


# ---------------------------------------------------------------------------
# Real mode — shells out to run_project.py. Requires API keys.
# ---------------------------------------------------------------------------


def run_real_arm(
    *,
    arm: str,
    project_dir: Path,
    run_project_args: list[str],
    seed_from_ara: str | None,
    seed_node_id: str | None,
    dry_run: bool,
) -> ArmResult:
    cmd = [sys.executable, "run_project.py", "--project-dir", str(project_dir)]
    if seed_from_ara:
        cmd.extend(["--seed-from-ara", seed_from_ara])
    if seed_node_id:
        cmd.extend(["--seed-node-id", seed_node_id])
    cmd.extend(run_project_args)

    started_at = _now_iso()
    started_perf = time.perf_counter()
    exit_code = 0
    if dry_run:
        notes = [f"dry-run cmd: {' '.join(cmd)}"]
    else:
        completed = subprocess.run(cmd, check=False)
        exit_code = completed.returncode
        notes = []
    finished_perf = time.perf_counter()

    ara_summary = _summarise_latest_ara(project_dir) if not dry_run else {}
    return ArmResult(
        arm=arm,
        started_at=started_at,
        finished_at=_now_iso(),
        duration_seconds=finished_perf - started_perf,
        llm_calls=-1,  # real mode: not directly instrumented (see notes)
        used_seed=bool(seed_from_ara),
        exit_code=exit_code,
        ara_root=ara_summary.get("ara_root"),
        node_count=ara_summary.get("node_count"),
        buggy_ratio=ara_summary.get("buggy_ratio"),
        node_content_hashes=ara_summary.get("hashes") or [],
        notes=notes + ara_summary.get("notes", []),
    )


def _summarise_latest_ara(project_dir: Path) -> dict[str, Any]:
    """Peek at the newest ARA under <project_dir>/ara/ and pull key counts."""
    from ai_scientist.utils.ara_artifact import iter_ara_exports

    manifests = list(iter_ara_exports(project_dir))
    if not manifests:
        return {"notes": ["no ARA export found after real-mode run"]}
    latest = max(manifests, key=lambda p: p.stat().st_mtime)
    try:
        manifest_payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"notes": [f"failed to read {latest}"]}
    ara_root = latest.parent
    graph_path = ara_root / "exploration_graph.json"
    hashes: list[str] = []
    buggy_ratio: float | None = None
    if graph_path.exists():
        try:
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            nodes = graph.get("nodes") or []
            hashes = [n.get("content_hash") for n in nodes if n.get("content_hash")]
            if nodes:
                buggy = sum(1 for n in nodes if n.get("is_buggy"))
                buggy_ratio = buggy / len(nodes)
        except (OSError, json.JSONDecodeError):
            hashes = []
    return {
        "ara_root": str(ara_root),
        "node_count": manifest_payload.get("counts", {}).get("nodes"),
        "buggy_ratio": buggy_ratio,
        "hashes": hashes,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def compute_hash_overlap(baseline: ArmResult | None, ara_seed: ArmResult | None) -> dict[str, Any]:
    if not baseline or not ara_seed:
        return {}
    b = set(baseline.node_content_hashes or [])
    s = set(ara_seed.node_content_hashes or [])
    inter = b & s
    return {
        "baseline_hashes": len(b),
        "ara_seed_hashes": len(s),
        "shared": sorted(inter),
        "shared_count": len(inter),
        "jaccard": (len(inter) / len(b | s)) if (b or s) else None,
    }


def _speedup(a: ArmResult, b: ArmResult) -> float | None:
    """Return baseline_time / seed_time. >1 means the seed was faster."""
    if a.duration_seconds <= 0 or b.duration_seconds <= 0:
        return None
    return a.duration_seconds / b.duration_seconds


def build_verdict(baseline: ArmResult, ara_seed: ArmResult) -> dict[str, Any]:
    speedup = _speedup(baseline, ara_seed)
    saved_calls = None
    if baseline.llm_calls >= 0 and ara_seed.llm_calls >= 0:
        saved_calls = baseline.llm_calls - ara_seed.llm_calls
    return {
        "seed_short_circuited": ara_seed.used_seed,
        "llm_calls_saved_on_draft": saved_calls,
        "wall_clock_speedup": speedup,
        "conclusion": _conclusion(baseline, ara_seed, speedup, saved_calls),
    }


def _conclusion(baseline: ArmResult, ara_seed: ArmResult, speedup, saved_calls) -> str:
    if not ara_seed.used_seed:
        return "seed_did_not_short_circuit"
    if saved_calls is not None and saved_calls > 0:
        return "seed_saved_llm_calls"
    if speedup is not None and speedup > 1.05:
        return "seed_wall_clock_faster"
    return "seed_inconclusive"


def write_report(report: ABReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ab_report.json"
    path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def render_console_summary(report: ABReport) -> str:
    lines = ["# ARA A/B report", f"mode: {report.mode}", ""]
    for label, arm in (("baseline", report.baseline), ("ara_seed", report.ara_seed)):
        if arm is None:
            lines.append(f"{label}: (missing)")
            continue
        lines.append(
            f"{label}: duration={arm.duration_seconds:.3f}s "
            f"llm_calls={arm.llm_calls} used_seed={arm.used_seed} "
            f"nodes={arm.node_count} exit={arm.exit_code}"
        )
    if report.verdict:
        lines.append("")
        lines.append(f"verdict: {report.verdict}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


def run_ab_stub(
    *,
    seed_manifest_path: Path,
    out_dir: Path,
    idea_hint: str | None = None,
) -> ABReport:
    """CI-safe A/B: exercise the seed short-circuit twice and diff."""
    baseline = run_stub_arm(arm="baseline", seed_manifest_path=None)
    seeded = run_stub_arm(arm="ara_seed", seed_manifest_path=seed_manifest_path)
    report = ABReport(
        mode="stub",
        idea_hint=idea_hint,
        baseline=baseline,
        ara_seed=seeded,
        hash_overlap=compute_hash_overlap(baseline, seeded),
        verdict=build_verdict(baseline, seeded),
    )
    write_report(report, out_dir)
    return report


def run_ab_real(
    *,
    project_dir_baseline: Path,
    project_dir_seeded: Path,
    seed_from_ara: str,
    seed_node_id: str | None,
    run_project_args: list[str],
    out_dir: Path,
    dry_run: bool = False,
    idea_hint: str | None = None,
) -> ABReport:
    baseline = run_real_arm(
        arm="baseline",
        project_dir=project_dir_baseline,
        run_project_args=run_project_args,
        seed_from_ara=None,
        seed_node_id=None,
        dry_run=dry_run,
    )
    seeded = run_real_arm(
        arm="ara_seed",
        project_dir=project_dir_seeded,
        run_project_args=run_project_args,
        seed_from_ara=seed_from_ara,
        seed_node_id=seed_node_id,
        dry_run=dry_run,
    )
    report = ABReport(
        mode="real_dry_run" if dry_run else "real",
        idea_hint=idea_hint,
        baseline=baseline,
        ara_seed=seeded,
        hash_overlap=compute_hash_overlap(baseline, seeded),
        verdict=build_verdict(baseline, seeded),
    )
    write_report(report, out_dir)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ara-ab",
        description="Run an A/B comparison between an ARA-seeded and cold-start XScientist arm.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    stub = sub.add_parser("stub", help="CI-safe stub. No API keys required.")
    stub.add_argument("--seed-manifest", required=True, help="Path to a staged ara_seed.json")
    stub.add_argument("--out-dir", required=True)
    stub.add_argument("--idea-hint", default=None)
    stub.set_defaults(func=_cmd_stub)

    real = sub.add_parser("real", help="Shell out to run_project.py twice. Requires API keys.")
    real.add_argument("--project-dir-baseline", required=True)
    real.add_argument("--project-dir-seeded", required=True)
    real.add_argument("--seed-from-ara", required=True)
    real.add_argument("--seed-node-id", default=None)
    real.add_argument("--out-dir", required=True)
    real.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    real.add_argument("--idea-hint", default=None)
    real.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to run_project.py (place after `--`).",
    )
    real.set_defaults(func=_cmd_real)

    return parser


def _cmd_stub(args: argparse.Namespace) -> int:
    report = run_ab_stub(
        seed_manifest_path=Path(args.seed_manifest).expanduser(),
        out_dir=Path(args.out_dir).expanduser(),
        idea_hint=args.idea_hint,
    )
    print(render_console_summary(report))
    return 0 if (report.ara_seed and report.ara_seed.used_seed) else 1


def _cmd_real(args: argparse.Namespace) -> int:
    # Strip a leading `--` if the user separated forwarded args with one.
    forwarded = list(args.extra or [])
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    report = run_ab_real(
        project_dir_baseline=Path(args.project_dir_baseline).expanduser(),
        project_dir_seeded=Path(args.project_dir_seeded).expanduser(),
        seed_from_ara=args.seed_from_ara,
        seed_node_id=args.seed_node_id,
        run_project_args=forwarded,
        out_dir=Path(args.out_dir).expanduser(),
        dry_run=bool(args.dry_run),
        idea_hint=args.idea_hint,
    )
    print(render_console_summary(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
