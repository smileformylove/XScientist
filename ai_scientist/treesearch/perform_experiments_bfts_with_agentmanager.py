from __future__ import annotations
import atexit
from datetime import datetime
import logging
import math
import shutil
import json
import pickle
import statistics
from . import backend
from .journal import Journal, Node
from .journal2report import journal2report
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.status import Status
from rich.tree import Tree
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from .agent_manager import AgentManager
from pathlib import Path
from .agent_manager import Stage
from .log_summarization import overall_summarize


logger = logging.getLogger("ai-scientist")


def journal_to_rich_tree(journal: Journal, cfg):
    best_node = journal.get_best_node_by_metric()

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def perform_experiments_bfts(config_path: str):
    # turn config path string into a path object
    config_path = Path(config_path)
    cfg = load_cfg(config_path)
    logger.info(f'Starting run "{cfg.exp_name}"')

    task_desc = load_task_desc(cfg)
    print(task_desc)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    global_step = 0

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    results_tsv_path = cfg.log_dir / "results.tsv"
    program_md_path = cfg.log_dir / "program.md"

    if not results_tsv_path.exists():
        results_tsv_path.write_text(
            "\t".join(
                [
                    "time",
                    "stage",
                    "step",
                    "kind",
                    "node_id",
                    "parent_id",
                    "status",
                    "decision",
                    "objective",
                    "metric_mean",
                    "metric_name",
                    "maximize",
                    "datasets",
                    "exec_time_sec",
                    "loc",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    logged_node_ids: set[str] = set()
    stage_best: dict[str, dict] = {}

    def _metric_meta(node: Node) -> tuple[float | None, float | None, str | None, bool | None]:
        metric = getattr(node, "metric", None)
        if metric is None or getattr(metric, "value", None) is None:
            return None, None, None, None
        try:
            metric_mean = float(metric.get_mean_value())
        except Exception:
            return None, None, None, None
        if math.isnan(metric_mean) or math.isinf(metric_mean):
            return None, None, None, None
        try:
            maximize = bool(metric._should_maximize())
        except Exception:
            maximize = None

        metric_name = getattr(metric, "name", None)
        value = getattr(metric, "value", None)
        if metric_name is None and isinstance(value, dict) and "metric_names" in value:
            try:
                metric_name = (value.get("metric_names") or [])[0].get("metric_name")
            except Exception:
                metric_name = None

        objective = metric_mean if (maximize is True or maximize is None) else -metric_mean
        return metric_mean, objective, metric_name, maximize

    def _write_program_md(stage: Stage, journal: Journal) -> None:
        best_node = journal.get_best_node_by_metric()
        if best_node is None:
            best_node_id = None
            metric_mean = None
            objective = None
            metric_name = None
            maximize = None
            datasets = []
            seed_stats = None
        else:
            best_node_id = best_node.id
            metric_mean, objective, metric_name, maximize = _metric_meta(best_node)
            datasets = [
                ds for ds in (best_node.datasets_successfully_tested or []) if ds
            ]

            seed_values = []
            for node in journal.nodes:
                if not getattr(node, "is_seed_node", False):
                    continue
                if node.parent is None or node.parent.id != best_node.id:
                    continue
                if getattr(node, "is_buggy", False):
                    continue
                seed_mean, seed_obj, _, _ = _metric_meta(node)
                if seed_mean is None:
                    continue
                seed_values.append(seed_mean)
            seed_stats = (
                {
                    "count": len(seed_values),
                    "mean": statistics.mean(seed_values),
                    "stdev": statistics.pstdev(seed_values) if len(seed_values) > 1 else 0.0,
                }
                if seed_values
                else None
            )

        levers = [
            "hyperparameters (lr, batch size, epochs, optimizer)",
            "model architecture (stage 3)",
            "regularization and data augmentation",
            "dataset coverage (must satisfy stage requirements)",
            "runtime scaling within timeout budget",
        ]

        keep = [
            "objective improves vs stage-best (epsilon) while keeping evaluation comparable",
            "meets stage-specific validation constraints (e.g. >=2 datasets in stage 2)",
            "passes without crashes/timeouts and produces valid metrics",
        ]
        discard = [
            "crash/timeout/no valid metrics",
            "objective regression or no meaningful improvement",
            "fails dataset coverage constraints",
        ]

        lines = [
            "# Experiment Autoresearch Program",
            "",
            f"- Generated at: {datetime.now().isoformat()}",
            f"- Run name: {cfg.exp_name}",
            f"- Workspace: {cfg.workspace_dir}",
            f"- Log dir: {cfg.log_dir}",
            f"- Current stage: {stage.name}",
            f"- Time budget per execution: {cfg.exec.timeout} sec",
            "",
            "## Fixed Evaluation Harness",
            "- Primary objective is computed from the parsed primary metric (mean across datasets).",
            "- Objective direction is derived from `lower_is_better` when available; otherwise defaults to maximize.",
            "- Multi-seed re-evaluation is used for stability checks when available.",
            "",
            "## Current Best (This Stage)",
            f"- best_node_id: {best_node_id}",
            f"- metric_name: {metric_name}",
            f"- metric_mean: {metric_mean}",
            f"- objective: {objective}",
            f"- maximize: {maximize}",
            f"- datasets: {datasets}",
            f"- seed_eval: {seed_stats}",
            "",
            "## Keep Criteria",
            *[f"- {item}" for item in keep],
            "",
            "## Discard Criteria",
            *[f"- {item}" for item in discard],
            "",
            "## Adjustable Levers",
            *[f"- {item}" for item in levers],
            "",
            "## Artifacts",
            f"- results.tsv: {results_tsv_path}",
            "- stage_*/notes/stage_progress.json: per-step progress snapshots",
        ]
        program_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    manager = AgentManager(
        task_desc=task_desc,
        cfg=cfg,
        workspace_dir=Path(cfg.workspace_dir),
    )

    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    status = Status("[green]Running experiments...")
    prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

    def create_exec_callback(status_obj):
        def exec_callback(*args, **kwargs):
            status_obj.update("[magenta]Executing code...")
            res = interpreter.run(*args, **kwargs)
            status_obj.update("[green]Generating code...")
            return res

        return exec_callback

    def step_callback(stage, journal):
        print("Step complete")
        try:
            # Generate and save notes for this step
            notes_dir = cfg.log_dir / f"stage_{stage.name}" / "notes"
            notes_dir.mkdir(parents=True, exist_ok=True)

            # Save latest node summary
            if journal.nodes:
                latest_node = journal.nodes[-1]
                if hasattr(latest_node, "_agent"):
                    summary = latest_node._agent._generate_node_summary(latest_node)
                    with open(
                        notes_dir / f"node_{latest_node.id}_summary.json", "w"
                    ) as f:
                        json.dump(summary, f, indent=2)


            if cfg.agent.get("summary", None) is not None:
                current_findings = journal.generate_summary(
                    include_code=False, 
                    **{
                        "model": cfg.agent.summary.model, 
                        "temp": cfg.agent.summary.temp
                    }
                )
            else:
                current_findings = journal.generate_summary(include_code=False)

            best_node = journal.get_best_node_by_metric()
            best_metric = best_node.metric if best_node else None

            best_metric_mean = None
            best_metric_objective = None
            if best_metric is not None:
                try:
                    best_metric_mean = float(best_metric.get_mean_value())
                    best_metric_objective = (
                        best_metric_mean
                        if best_metric._should_maximize()
                        else -best_metric_mean
                    )
                except Exception:
                    best_metric_mean = None
                    best_metric_objective = None

            seed_stats = None
            if best_node is not None:
                seed_values = []
                for node in journal.nodes:
                    if not getattr(node, "is_seed_node", False):
                        continue
                    if node.parent is None or node.parent.id != best_node.id:
                        continue
                    if getattr(node, "is_buggy", False):
                        continue
                    metric = getattr(node, "metric", None)
                    if metric is None:
                        continue
                    seed_mean, _, _, _ = _metric_meta(node)
                    if seed_mean is None:
                        continue
                    seed_values.append(seed_mean)

                if seed_values:
                    seed_stats = {
                        "count": len(seed_values),
                        "mean": statistics.mean(seed_values),
                        "stdev": statistics.pstdev(seed_values) if len(seed_values) > 1 else 0.0,
                        "values": seed_values[:10],
                    }

            # Generate and save stage progress summary
            stage_summary = {
                "stage": stage.name,
                "total_nodes": len(journal.nodes),
                "buggy_nodes": len(journal.buggy_nodes),
                "good_nodes": len(journal.good_nodes),
                "best_node_id": best_node.id if best_node else None,
                "best_metric": str(best_metric) if best_metric else "None",
                "best_metric_mean": best_metric_mean,
                "best_metric_objective": best_metric_objective,
                "datasets_successfully_tested": (
                    best_node.datasets_successfully_tested if best_node else []
                ),
                "seed_eval": seed_stats,
                "current_findings": current_findings,
            }

            with open(notes_dir / "stage_progress.json", "w") as f:
                json.dump(stage_summary, f, indent=2)

            # Update autoresearch-style program snapshot and results ledger.
            _write_program_md(stage, journal)
            now = datetime.now().isoformat()

            stage_key = stage.name
            stage_best_entry = stage_best.get(stage_key)
            for node in journal.nodes:
                node_id = getattr(node, "id", None)
                if not node_id or node_id in logged_node_ids:
                    continue
                logged_node_ids.add(node_id)

                is_seed = bool(getattr(node, "is_seed_node", False))
                is_seed_agg = bool(getattr(node, "is_seed_agg_node", False))
                kind = "seed_agg" if is_seed_agg else ("seed" if is_seed else "main")
                parent_id = node.parent.id if getattr(node, "parent", None) else ""

                loc = len((node.code or "").splitlines())
                datasets = [
                    ds for ds in (getattr(node, "datasets_successfully_tested", None) or []) if ds
                ]

                metric_mean, objective, metric_name, maximize = _metric_meta(node)
                status = (
                    "ok"
                    if (node.is_buggy is False and node.is_buggy_plots is False and objective is not None)
                    else "crash"
                )

                decision = ""
                if kind != "main":
                    decision = kind
                elif status != "ok":
                    decision = "discard"
                else:
                    assert objective is not None
                    # Stage-specific validation constraints (mirrors AgentManager gating).
                    if getattr(stage, "stage_number", None) == 2 and len(set(datasets)) < 2:
                        status = "invalid"
                        decision = "discard"
                        # Do not allow "keep" promotion if dataset coverage is insufficient.
                        objective_for_keep = None
                    else:
                        objective_for_keep = objective

                    if objective_for_keep is None:
                        # Skip keep/discard evaluation beyond marking discard.
                        pass
                    else:
                        # Autoresearch-style keep/discard based on objective improvement; break ties by simplicity.
                        eps = (
                            max(1e-6, 1e-3 * abs(stage_best_entry["objective"]))
                            if stage_best_entry
                            else 1e-6
                        )
                        keep = False
                        if stage_best_entry is None:
                            keep = True
                        elif objective_for_keep > stage_best_entry["objective"] + eps:
                            keep = True
                        elif (
                            abs(objective_for_keep - stage_best_entry["objective"])
                            <= eps
                            and loc < stage_best_entry.get("loc", loc)
                        ):
                            keep = True

                        decision = "keep" if keep else "discard"
                        if keep:
                            stage_best_entry = {
                                "objective": objective_for_keep,
                                "loc": loc,
                                "node_id": node_id,
                            }
                            stage_best[stage_key] = stage_best_entry

                with open(results_tsv_path, "a", encoding="utf-8") as handle:
                    handle.write(
                        "\t".join(
                            [
                                now,
                                stage.name,
                                str(getattr(node, "step", "")),
                                kind,
                                node_id,
                                parent_id,
                                status,
                                decision,
                                "" if objective is None else f"{objective:.12g}",
                                "" if metric_mean is None else f"{metric_mean:.12g}",
                                str(metric_name or ""),
                                "" if maximize is None else str(bool(maximize)),
                                ",".join(datasets),
                                "" if getattr(node, "exec_time", None) is None else f"{float(node.exec_time):.6f}",
                                str(loc),
                            ]
                        )
                        + "\n"
                    )

            # Save the run as before
            save_run(cfg, journal, stage_name=f"stage_{stage.name}")

        except Exception as e:
            print(f"Error in step callback: {e}")

        print(f"Run saved at {cfg.log_dir / f'stage_{stage.name}'}")
        print(f"Step {len(journal)}/{stage.max_iterations} at stage_{stage.name}")
        print(f"Run saved at {cfg.log_dir / f'stage_{stage.name}'}")

    def generate_live(manager):
        current_stage = manager.current_stage
        current_journal = manager.journals.get(
            current_stage.name if current_stage else None, None
        )

        if current_journal:
            tree = journal_to_rich_tree(current_journal, cfg)
        else:
            tree = Tree("[bold blue]No results yet")

        file_paths = [
            f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
            f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
            f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
        ]

        stage_info = [
            "[bold]Experiment Progress:",
            f"Current Stage: [cyan]{current_stage.name if current_stage else 'None'}[/cyan]",
            f"Completed Stages: [green]{', '.join(manager.completed_stages)}[/green]",
        ]

        left = Group(
            Panel(Text(task_desc_str.strip()), title="Task description"),
            Panel(Text("\n".join(stage_info)), title="Stage Progress"),
            prog,
            status,
        )
        right = tree
        wide = Group(*file_paths)

        return Panel(
            Group(
                Padding(wide, (1, 1, 1, 1)),
                Columns(
                    [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                    equal=True,
                ),
            ),
            title=f'[b]AIDE is working on experiment: [bold green]"{cfg.exp_name}[/b]"',
            subtitle="Press [b]Ctrl+C[/b] to stop the run",
        )

    live = Live(
        generate_live(manager),
        refresh_per_second=16,
        screen=True,
    )

    manager.run(exec_callback=create_exec_callback(status), step_callback=step_callback)

    manager_pickle_path = cfg.log_dir / "manager.pkl"
    try:
        with open(manager_pickle_path, "wb") as f:
            pickle.dump(manager, f)
        logger.info(f"Saved manager state to: {manager_pickle_path}")
    except Exception as e:
        logger.warning(f"Failed to save full manager state: {e}")
        try:
            with open(manager_pickle_path, "wb") as f:
                pickle.dump(manager.journals.items(), f)
            logger.info(f"Saved manager journals to: {manager_pickle_path}")
        except Exception as e:
            logger.error(f"Failed to save manager journals: {e}")

    if cfg.generate_report:
        print("Generating final report from all stages...")
        (
            draft_summary,
            baseline_summary,
            research_summary,
            ablation_summary,
        ) = overall_summarize(manager.journals.items(), cfg)
        draft_summary_path = cfg.log_dir / "draft_summary.json"
        baseline_summary_path = cfg.log_dir / "baseline_summary.json"
        research_summary_path = cfg.log_dir / "research_summary.json"
        ablation_summary_path = cfg.log_dir / "ablation_summary.json"

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


if __name__ == "__main__":
    cfg_path = "treesearch/utils/config.yaml"
    cfg = load_cfg(cfg_path)
    perform_experiments_bfts(cfg_path)
