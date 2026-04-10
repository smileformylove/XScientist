from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ai_scientist.utils.pipeline_helpers import find_latest_bfts_run_dir


def _safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _metric_should_maximize(metric: dict) -> bool:
    maximize = metric.get("maximize")
    if isinstance(maximize, bool):
        return maximize
    value = metric.get("value")
    if isinstance(value, dict) and "metric_names" in value:
        try:
            first = (value.get("metric_names") or [])[0]
            lower_is_better = bool(first.get("lower_is_better"))
            return not lower_is_better
        except Exception:
            return True
    return True


def _metric_mean_value(metric: dict) -> Optional[float]:
    value = metric.get("value")
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return _safe_float(value)

    if isinstance(value, dict):
        if "metric_names" in value:
            all_values: list[float] = []
            for metric_entry in value.get("metric_names") or []:
                for point in metric_entry.get("data") or []:
                    final_value = _safe_float(point.get("final_value"))
                    if final_value is not None:
                        all_values.append(final_value)
            return statistics.mean(all_values) if all_values else None

        # legacy: dataset -> value dict
        all_values = []
        for raw in value.values():
            parsed = _safe_float(raw)
            if parsed is not None:
                all_values.append(parsed)
        return statistics.mean(all_values) if all_values else None

    return None


def _metric_objective(metric: dict) -> Optional[float]:
    mean_value = _metric_mean_value(metric)
    if mean_value is None:
        return None
    return mean_value if _metric_should_maximize(metric) else -mean_value


def _extract_dataset_names(metric: dict) -> list[str]:
    value = metric.get("value")
    names = set()
    if isinstance(value, dict):
        if "metric_names" in value:
            for metric_entry in value.get("metric_names") or []:
                for point in metric_entry.get("data") or []:
                    ds = point.get("dataset_name")
                    if ds:
                        names.add(str(ds).strip())
        else:
            for key in value.keys():
                if key:
                    names.add(str(key).strip())
    return sorted(name for name in names if name)


def _stage_sort_key(stage_dir: Path) -> tuple[int, str]:
    name = stage_dir.name
    match = re.search(r"stage_(\d+)_", name)
    stage_num = int(match.group(1)) if match else 999
    return stage_num, name


@dataclass
class StageBest:
    stage: str
    journal_path: str
    best_node_id: Optional[str]
    best_metric_name: Optional[str]
    best_metric_mean: Optional[float]
    best_metric_objective: Optional[float]
    dataset_names: list[str]
    seed_eval: Optional[dict]
    node_counts: dict


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def build_experiment_report(base_folder: str | Path) -> dict[str, Any]:
    base_folder = Path(base_folder)
    run_dir = find_latest_bfts_run_dir(base_folder, logs_subdir="logs")
    if run_dir is None:
        return {
            "base_folder": str(base_folder),
            "latest_run_dir": None,
            "stages": [],
            "warnings": ["No BFTS run directory found under logs/."],
        }

    stage_dirs = sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("stage_")],
        key=_stage_sort_key,
    )
    warnings: list[str] = []
    stage_reports: list[dict[str, Any]] = []
    prev_objective: Optional[float] = None

    for stage_dir in stage_dirs:
        journal_path = stage_dir / "journal.json"
        journal_json = _load_json(journal_path)
        nodes = journal_json.get("nodes") or []
        node2parent = journal_json.get("node2parent") or {}

        def _is_good(node: dict) -> bool:
            return (node.get("is_buggy") is False) and (node.get("is_buggy_plots") is False)

        candidates = [
            node
            for node in nodes
            if _is_good(node)
            and not node.get("is_seed_node", False)
            and not node.get("is_seed_agg_node", False)
        ]
        best_node = None
        best_objective = None
        for node in candidates:
            metric = node.get("metric") or {}
            objective = _metric_objective(metric)
            if objective is None:
                continue
            if best_objective is None or objective > best_objective:
                best_objective = objective
                best_node = node

        if best_node is None:
            warnings.append(f"{stage_dir.name}: no valid non-seed nodes with metrics found.")
            stage_reports.append(
                {
                    "stage_dir": str(stage_dir),
                    "journal_path": str(journal_path) if journal_path.exists() else None,
                    "best": None,
                    "delta_objective_vs_prev_stage": None,
                    "node_counts": {
                        "total": len(nodes),
                        "good": sum(1 for n in nodes if _is_good(n)),
                        "buggy": sum(1 for n in nodes if n.get("is_buggy") is True),
                    },
                }
            )
            continue

        metric = best_node.get("metric") or {}
        best_mean = _metric_mean_value(metric)
        best_name = metric.get("name")
        metric_value = metric.get("value") or {}
        if best_name is None and isinstance(metric_value, dict) and "metric_names" in metric_value:
            try:
                best_name = (metric_value.get("metric_names") or [])[0].get("metric_name")
            except Exception:
                best_name = None

        dataset_names = _extract_dataset_names(metric)

        # Seed stats for the best node (if any)
        seed_values: list[float] = []
        best_id = best_node.get("id")
        if best_id:
            for node in nodes:
                if not _is_good(node):
                    continue
                if not node.get("is_seed_node", False):
                    continue
                parent_id = node2parent.get(node.get("id"))
                if parent_id != best_id:
                    continue
                seed_metric = node.get("metric") or {}
                seed_mean = _metric_mean_value(seed_metric)
                if seed_mean is not None:
                    seed_values.append(seed_mean)

        seed_eval = None
        if seed_values:
            seed_eval = {
                "count": len(seed_values),
                "mean": statistics.mean(seed_values),
                "stdev": statistics.pstdev(seed_values) if len(seed_values) > 1 else 0.0,
                "values": seed_values[:10],
            }

        delta = None
        if prev_objective is not None and best_objective is not None:
            delta = best_objective - prev_objective
        prev_objective = best_objective

        stage_reports.append(
            {
                "stage_dir": str(stage_dir),
                "journal_path": str(journal_path) if journal_path.exists() else None,
                "best": {
                    "node_id": best_id,
                    "metric_name": best_name,
                    "metric_mean": best_mean,
                    "metric_objective": best_objective,
                    "dataset_names": dataset_names,
                    "seed_eval": seed_eval,
                },
                "delta_objective_vs_prev_stage": delta,
                "node_counts": {
                    "total": len(nodes),
                    "good": sum(1 for n in nodes if _is_good(n)),
                    "buggy": sum(1 for n in nodes if n.get("is_buggy") is True),
                },
            }
        )

    report: dict[str, Any] = {
        "base_folder": str(base_folder),
        "latest_run_dir": str(run_dir),
        "stages": stage_reports,
        "warnings": warnings,
    }

    # Optional, deterministic rigor heuristic.
    try:
        from ai_scientist.utils.high_quality_pipeline import assess_experiment_rigor

        report["rigor_heuristic"] = assess_experiment_rigor(base_folder)
    except Exception:
        report["rigor_heuristic"] = None

    return report


def render_experiment_report_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Experiment Report",
        "",
        f"- Base folder: {report.get('base_folder')}",
        f"- Latest run dir: {report.get('latest_run_dir')}",
        "",
    ]

    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["## Warnings", ""])
        for item in warnings:
            lines.append(f"- {item}")
        lines.append("")

    rigor = report.get("rigor_heuristic")
    if isinstance(rigor, dict):
        lines.extend(
            [
                "## Rigor Heuristic",
                "",
                f"- Score: {rigor.get('score')}",
                f"- Assessment: {rigor.get('overall_assessment')}",
                "",
            ]
        )
        recs = rigor.get("recommendations") or []
        if recs:
            lines.append("### Recommendations")
            lines.append("")
            for rec in recs[:8]:
                lines.append(f"- {rec}")
            lines.append("")

    lines.extend(["## Stage Summary", ""])
    for stage in report.get("stages") or []:
        best = stage.get("best") or {}
        lines.append(f"### {Path(stage.get('stage_dir') or '').name}")
        lines.append("")
        lines.append(f"- Nodes: total={stage.get('node_counts', {}).get('total')}, good={stage.get('node_counts', {}).get('good')}, buggy={stage.get('node_counts', {}).get('buggy')}")
        if stage.get("delta_objective_vs_prev_stage") is not None:
            lines.append(f"- Delta objective vs prev stage: {stage.get('delta_objective_vs_prev_stage'):+.6f}")
        if best:
            lines.append(f"- Best node: {best.get('node_id')}")
            lines.append(f"- Metric: {best.get('metric_name')} mean={best.get('metric_mean')} objective={best.get('metric_objective')}")
            ds = best.get("dataset_names") or []
            lines.append(f"- Datasets ({len(ds)}): {', '.join(ds) if ds else 'N/A'}")
            seed = best.get("seed_eval")
            if isinstance(seed, dict):
                lines.append(f"- Seed eval: n={seed.get('count')} mean={seed.get('mean')} stdev={seed.get('stdev')}")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_experiment_report(base_folder: str | Path) -> tuple[Path, Path]:
    base_folder = Path(base_folder)
    report = build_experiment_report(base_folder)
    json_path = base_folder / "experiment_report.json"
    md_path = base_folder / "experiment_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_experiment_report_markdown(report), encoding="utf-8")
    return json_path, md_path

