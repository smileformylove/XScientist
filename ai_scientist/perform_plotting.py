# Modified by XScientist contributors from the AI-Scientist-v2/AIDE lineage.
# See THIRD_PARTY_NOTICES.md for provenance and license details.
import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from rich import print

from ai_scientist.llm import create_client, get_response_from_llm
from ai_scientist.utils.figure_spec import (
    build_figure_spec_from_summaries,
    render_figure_spec_markdown,
    save_figure_spec,
)
from ai_scientist.utils.pipeline_contracts import load_contract_artifact
from ai_scientist.utils.token_tracker import token_tracker
from ai_scientist.utils.auth_session import require_login
from ai_scientist.utils.atomic_io import atomic_write_bytes, atomic_write_json
from ai_scientist.utils.safe_files import BoundedFileError, read_bounded_regular_file
from ai_scientist.treesearch.interpreter import (
    Interpreter,
    SandboxPolicy,
    sandbox_policy_from_config,
)
from ai_scientist.perform_icbinb_writeup import (
    load_idea_text,
    load_exp_summaries,
    filter_experiment_summaries,
)

MAX_FIGURES = 12
MAX_AGGREGATOR_CODE_BYTES = 2 * 1024 * 1024
MAX_FIGURE_BYTES = 50 * 1024 * 1024
MAX_FIGURE_DIMENSION = 20_000
MAX_FIGURE_PIXELS = 50_000_000
MAX_QUALITY_GUIDANCE_BYTES = 2 * 1024 * 1024
QUALITY_PLOT_GUIDANCE_FILES = (
    "experiment_visualization_brief.md",
    "experiment_analysis.md",
    "figure_caption_guidance.md",
    "table_caption_guidance.md",
    "architecture_figure_brief.md",
    "humanizer_style_notes.md",
)


class PlotAggregationError(RuntimeError):
    """Raised when a final plot set cannot be produced safely."""


class PlotExecutionPolicyError(PlotAggregationError):
    """Raised when generated plotting code lacks an acceptable execution boundary."""


@dataclass(frozen=True)
class PlotRunResult:
    output: str
    succeeded: bool
    exc_type: str | None
    execution_backend: str
    isolation: dict[str, Any]


@dataclass(frozen=True)
class PlotExecutionSettings:
    policy: SandboxPolicy
    timeout: int


AGGREGATOR_SYSTEM_MSG = f"""You are an ambitious AI researcher who is preparing final plots for a scientific paper submission.
You have multiple experiment summaries (baseline, research, ablation), each possibly containing references to different plots or numerical insights.
There is also a top-level 'research_idea.md' file that outlines the overarching research direction.
Your job is to produce ONE Python script that fully aggregates and visualizes the final results for a comprehensive research paper.

Key points:
1) Combine or replicate relevant existing plotting code, referencing how data was originally generated (from code references) to ensure correctness.
2) Create a complete set of final scientific plots, stored in 'figures/' only (since only those are used in the final paper).
3) Make sure to use existing .npy data for analysis; do NOT hallucinate data. If single numeric results are needed, these may be copied from the JSON summaries.
4) Only create plots where the data is best presented as a figure and not as a table. E.g. don't use bar plots if the data is hard to visually compare.
5) The final aggregator script must be in triple backticks and stand alone so it can be dropped into a codebase and run.
6) If there are plots based on synthetic data, include them in the appendix.

Implement best practices:
- Do not produce extraneous or irrelevant plots.
- Maintain clarity, minimal but sufficient code.
- Demonstrate thoroughness for a final research paper submission.
- Do NOT reference non-existent files or images.
- Use the .npy files to get data for the plots and key numbers from the JSON summaries.
- Demarcate each individual plot, and put them in separate try-catch blocks so that the failure of one plot does not affect the others.
- Make sure to only create plots that are unique and needed for the final paper and appendix. A good number could be around {MAX_FIGURES} plots in total.
- Aim to aggregate multiple figures into one plot if suitable, i.e. if they are all related to the same topic. You can place up to 3 plots in one row.
- Provide well-labeled plots (axes, legends, titles) that highlight main findings. Use informative names everywhere, including in the legend for referencing them in the final paper. Make sure the legend is always visible.
- Make the plots look professional (if applicable, no top and right spines, dpi of 300, adequate ylim, etc.).
- Do not use labels with underscores, e.g. "loss_vs_epoch" should be "loss vs epoch".
- For image examples, select a few categories/classes to showcase the diversity of results instead of showing a single category/class. Some can be included in the main paper, while the rest can go in the appendix.

Your output should be the entire Python aggregator script in triple backticks.
"""


def _quality_plot_guidance_paths(base_path: Path) -> list[Path]:
    quality_dir = base_path / "quality"
    try:
        metadata = quality_dir.lstat()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise PlotAggregationError("Could not inspect plot guidance directory") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PlotAggregationError("Plot guidance path must be a regular directory")
    return [
        path
        for filename in QUALITY_PLOT_GUIDANCE_FILES
        if (path := quality_dir / filename).exists() or path.is_symlink()
    ]


def load_quality_plot_guidance(base_folder: str) -> str:
    base_path = Path(base_folder).expanduser().resolve()
    sections = []
    for path in _quality_plot_guidance_paths(base_path):
        try:
            payload = read_bounded_regular_file(
                path,
                maximum=MAX_QUALITY_GUIDANCE_BYTES,
                label="plot_quality_guidance",
            )
            content = payload.decode("utf-8").strip()
        except (BoundedFileError, UnicodeDecodeError) as exc:
            raise PlotAggregationError(
                f"Plot guidance input is unsafe or invalid: {path.name}"
            ) from exc
        if not content:
            continue
        sections.append(f"--- {path.name} ---\n{content}")
    return "\n\n".join(sections)


def build_aggregator_prompt(
    combined_summaries_str,
    idea_text,
    quality_guidance_text,
    figure_spec_markdown,
):
    return f"""
We have three JSON summaries of scientific experiments: baseline, research, ablation.
They may contain lists of figure descriptions, code to generate the figures, and paths to the .npy files containing the numerical results.
Our goal is to produce final, publishable figures.

--- RESEARCH IDEA ---
```
{idea_text}
```

--- QUALITY / VISUALIZATION GUIDANCE ---
```
{quality_guidance_text or "No extra visualization guidance available."}
```

--- STRUCTURED FIGURE SPEC ---
```
{figure_spec_markdown or "No structured figure spec available."}
```

IMPORTANT:
- The aggregator script must load existing .npy experiment data from the "exp_results_npy_files" fields (ONLY using full and exact file paths in the summary JSONs) for thorough plotting.
- It should call os.makedirs("figures", exist_ok=True) before saving any plots.
- Aim for a balance of empirical results, ablations, and diverse, informative visuals in 'figures/' that comprehensively showcase the finalized research outcomes.
- If you need .npy paths from the summary, only copy those paths directly (rather than copying and parsing the entire summary).

Your generated Python script must:
1) Load or refer to relevant data and .npy files from these summaries. Use the full and exact file paths in the summary JSONs.
2) Synthesize or directly create final, scientifically meaningful plots for a final research paper (comprehensive and complete), referencing the original code if needed to see how the data was generated.
3) Carefully combine or replicate relevant existing plotting code to produce these final aggregated plots in 'figures/' only, since only those are used in the final paper.
4) Do not hallucinate data. Data must either be loaded from .npy files or copied from the JSON summaries.
5) The aggregator script must be fully self-contained, and place the final plots in 'figures/'.
6) This aggregator script should produce a comprehensive and final set of scientific plots for the final paper, reflecting all major findings from the experiment data.
7) Make sure that every plot is unique and not duplicated from the original plots. Delete any duplicate plots if necessary.
8) Each figure can have up to 3 subplots using fig, ax = plt.subplots(1, 3).
9) Use a font size larger than the default for plot labels and titles to ensure they are readable in the final PDF paper.
10) Respect the visualization guidance files when choosing which plots to keep, how to title them, and how to connect them back to the paper's main claims.


Below are the summaries in JSON:

{combined_summaries_str}

Respond with a Python script in triple backticks.
"""


def extract_code_snippet(text: str) -> str:
    """
    Look for a Python code block in triple backticks in the LLM response.
    Return only that code. If no code block is found, return the entire text.
    """
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[0].strip() if matches else text.strip()


def _plot_execution_policy(
    execution_config_path: str | os.PathLike[str] | None,
    *,
    allow_unisolated_local_model_code: bool,
) -> SandboxPolicy:
    return _plot_execution_settings(
        execution_config_path,
        allow_unisolated_local_model_code=allow_unisolated_local_model_code,
    ).policy


def _plot_execution_settings(
    execution_config_path: str | os.PathLike[str] | None,
    *,
    allow_unisolated_local_model_code: bool,
) -> PlotExecutionSettings:
    """Resolve a fail-closed policy for executing model-generated plot code.

    Autonomous callers provide the exact BFTS config used for the experiment so
    the plot runner inherits its versioned Docker image and resource limits.
    Without an explicit local-development opt-in, even a legacy config that
    allowed process fallback is upgraded to require isolation.
    """

    if execution_config_path is None:
        if allow_unisolated_local_model_code:
            policy = SandboxPolicy(backend="process", require_isolation=False)
        else:
            policy = SandboxPolicy(backend="auto", require_isolation=True)
        return PlotExecutionSettings(policy=policy, timeout=3600)

    config_path = Path(execution_config_path).expanduser().resolve()
    try:
        config = OmegaConf.load(config_path)
        if not OmegaConf.is_dict(config):
            raise TypeError("configuration root must be a mapping")
        exec_config: Any = config.get("exec")
        if not OmegaConf.is_dict(exec_config):
            raise TypeError("exec must be a mapping")
        policy = sandbox_policy_from_config(exec_config)
        raw_timeout = exec_config.get("timeout", 3600)
        if isinstance(raw_timeout, bool):
            raise TypeError("exec.timeout must be a positive integer")
        timeout = int(raw_timeout)
        if timeout <= 0:
            raise ValueError("exec.timeout must be a positive integer")
    except Exception as exc:
        raise PlotExecutionPolicyError(
            f"Could not load plotting execution policy from {config_path}: {exc}"
        ) from exc

    if allow_unisolated_local_model_code:
        if policy.require_isolation:
            raise PlotExecutionPolicyError(
                "The execution config requires isolation; the local unisolated "
                "development opt-in cannot downgrade it."
            )
        return PlotExecutionSettings(policy=policy, timeout=timeout)

    return PlotExecutionSettings(
        policy=replace(policy, require_isolation=True, network="none"),
        timeout=timeout,
    )


def _validate_aggregator_script_target(
    aggregator_script_path: str | os.PathLike[str],
    base_folder: str | os.PathLike[str],
    script_name: str,
) -> tuple[Path, Path]:
    base_path = Path(base_folder).expanduser().resolve()
    script_path = Path(aggregator_script_path).expanduser()
    if Path(script_name).name != script_name:
        raise PlotExecutionPolicyError("Aggregator script name must be a leaf filename")
    if script_path.parent.resolve() != base_path or script_path.name != script_name:
        raise PlotExecutionPolicyError(
            "Aggregator script target must be a direct child of the experiment folder"
        )
    if script_path.is_symlink():
        raise PlotExecutionPolicyError(
            "Refusing to write model-generated plotting code through a symlink"
        )
    return base_path, script_path


def run_aggregator_script(
    aggregator_code,
    aggregator_script_path,
    base_folder,
    script_name,
    *,
    sandbox_policy: SandboxPolicy | None = None,
    timeout: int = 3600,
):
    if not aggregator_code.strip():
        print("No aggregator code was provided. Skipping aggregator script run.")
        raise PlotAggregationError("No aggregator code was provided")
    if len(aggregator_code.encode("utf-8")) > MAX_AGGREGATOR_CODE_BYTES:
        raise PlotAggregationError("Generated plotting script exceeds the code limit")
    base_path, script_path = _validate_aggregator_script_target(
        aggregator_script_path,
        base_folder,
        script_name,
    )
    policy = sandbox_policy or SandboxPolicy(backend="auto", require_isolation=True)

    try:
        interpreter = Interpreter(
            working_dir=base_path,
            agent_file_name=script_name,
            timeout=timeout,
            env_vars={},
            sandbox_policy=policy,
        )
    except Exception as exc:
        raise PlotExecutionPolicyError(
            "Generated plotting code executor is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    print(
        f"Aggregator script target is '{script_path}'. Executing with "
        f"backend={interpreter.execution_backend!r}, "
        f"isolated={interpreter.execution_backend == 'docker'}."
    )
    if interpreter.execution_backend != "docker":
        print(
            "[bold yellow]UNISOLATED LOCAL DEVELOPMENT MODE:[/bold yellow] "
            "model-generated plotting code runs in a local process and can access "
            "the host filesystem outside a Docker boundary."
        )

    aggregator_out = ""
    try:
        result = interpreter.run(aggregator_code)
        aggregator_out = "".join(result.term_out)
        isolation = dict(result.isolation or {})
        if policy.require_isolation and (
            result.execution_backend != "docker" or not isolation.get("isolated")
        ):
            raise PlotExecutionPolicyError(
                "Generated plotting code did not return an isolated Docker receipt"
            )
        if result.exc_type in {
            "SandboxUnavailableError",
            "TimeoutError",
            "ResourceLimitError",
        }:
            raise PlotAggregationError(
                "Generated plotting code was stopped by the execution boundary: "
                f"{result.exc_type}"
            )
        succeeded = result.exc_type is None
        if succeeded:
            print("Aggregator script ran successfully.")
        else:
            print(
                "Aggregator script needs revision after execution error "
                f"({result.exc_type})."
            )
        return PlotRunResult(
            output=aggregator_out,
            succeeded=succeeded,
            exc_type=result.exc_type,
            execution_backend=result.execution_backend,
            isolation=isolation,
        )
    except PlotAggregationError:
        raise
    except Exception as exc:
        raise PlotAggregationError(
            "Generated plotting code could not be executed safely: "
            f"{type(exc).__name__}"
        ) from exc
    finally:
        interpreter.cleanup_session()


def _validate_plot_outputs(
    *,
    figures_dir: str | os.PathLike[str],
    aggregator_script_path: str | os.PathLike[str],
    expected_code: str,
) -> list[dict[str, object]]:
    """Validate model-created files before host-side write-up code can read them."""

    try:
        script_payload = read_bounded_regular_file(
            aggregator_script_path,
            maximum=MAX_AGGREGATOR_CODE_BYTES,
            label="plot_aggregator_script",
        )
    except BoundedFileError as exc:
        raise PlotAggregationError(
            f"Generated plotting script failed safety validation: {exc.reason}"
        ) from exc
    if script_payload != expected_code.encode("utf-8"):
        raise PlotAggregationError(
            "Generated plotting script changed itself during execution"
        )

    directory = Path(figures_dir)
    try:
        directory.lstat()
    except OSError as exc:
        raise PlotAggregationError(
            "Generated plotting code produced no figures"
        ) from exc
    if directory.is_symlink() or not directory.is_dir():
        raise PlotAggregationError("Generated figures path is not a regular directory")

    entries = sorted(directory.iterdir(), key=lambda path: path.name)
    if not entries:
        raise PlotAggregationError("Generated plotting code produced no figures")
    if len(entries) > MAX_FIGURES:
        raise PlotAggregationError(
            f"Generated plotting code produced more than {MAX_FIGURES} artifacts"
        )

    validated: list[dict[str, object]] = []
    for path in entries:
        if path.suffix.casefold() != ".png":
            raise PlotAggregationError(
                f"Generated figure artifact is not a PNG: {path.name}"
            )
        try:
            payload = read_bounded_regular_file(
                path,
                maximum=MAX_FIGURE_BYTES,
                label="plot_figure",
            )
        except BoundedFileError as exc:
            raise PlotAggregationError(
                f"Generated figure {path.name!r} failed safety validation: {exc.reason}"
            ) from exc
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise PlotAggregationError(
                f"Generated figure artifact is not a valid PNG: {path.name}"
            )
        if len(payload) < 24 or payload[12:16] != b"IHDR":
            raise PlotAggregationError(
                f"Generated figure artifact has no valid PNG header: {path.name}"
            )
        width, height = struct.unpack(">II", payload[16:24])
        if (
            width <= 0
            or height <= 0
            or width > MAX_FIGURE_DIMENSION
            or height > MAX_FIGURE_DIMENSION
            or width * height > MAX_FIGURE_PIXELS
        ):
            raise PlotAggregationError(
                f"Generated figure dimensions exceed safety limits: {path.name}"
            )
        validated.append(
            {
                "path": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return validated


def _clear_staged_figures(figures_dir: str | os.PathLike[str]) -> None:
    directory = Path(figures_dir)
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise PlotAggregationError(
            "Could not safely reset staged plotting outputs"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise PlotAggregationError(
            "Generated plotting code replaced the figures directory unsafely"
        )
    shutil.rmtree(directory)


def _link_staged_plot_inputs(base_path: Path, execution_path: Path) -> None:
    """Expose legacy relative evidence paths through the read-only base mount."""

    for name in ("experiment_results", "logs", "quality"):
        source = base_path / name
        try:
            metadata = source.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise PlotExecutionPolicyError(
                f"Could not inspect staged plot input {name!r}"
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise PlotExecutionPolicyError(
                f"Staged plot input {name!r} must be a regular directory"
            )
        (execution_path / name).symlink_to(source, target_is_directory=True)


def _link_referenced_plot_data(
    *,
    base_path: Path,
    execution_path: Path,
    summaries: object,
) -> None:
    for source in _referenced_plot_data_paths(summaries, base_path=base_path):
        relative = source.relative_to(base_path)
        target = execution_path / relative
        if target.exists():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
        except OSError as exc:
            raise PlotExecutionPolicyError(
                f"Could not stage referenced plot data {relative.as_posix()!r}"
            ) from exc


def _optional_regular_payload(
    path: Path,
    *,
    maximum: int,
    label: str,
) -> bytes | None:
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PlotAggregationError(f"Could not inspect existing {label}") from exc
    try:
        return read_bounded_regular_file(path, maximum=maximum, label=label)
    except BoundedFileError as exc:
        raise PlotAggregationError(
            f"Existing {label} failed safety validation: {exc.reason}"
        ) from exc


def _stable_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _hash_bounded_regular_file(
    path: Path,
    *,
    maximum: int,
    label: str,
) -> tuple[int, str]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PlotAggregationError(f"Could not inspect {label}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise PlotAggregationError(f"Unsafe {label}: symlink_rejected")
    if not stat.S_ISREG(metadata.st_mode):
        raise PlotAggregationError(f"Unsafe {label}: not_regular")
    if metadata.st_size > maximum:
        raise PlotAggregationError(f"Oversized {label}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PlotAggregationError(f"Could not safely open {label}") from exc
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if _stable_metadata(before) != _stable_metadata(metadata):
            raise PlotAggregationError(f"{label} changed before hashing")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise PlotAggregationError(f"Oversized {label}")
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stable_metadata(after) != _stable_metadata(before) or total != after.st_size:
        raise PlotAggregationError(f"{label} changed while hashing")
    return total, digest.hexdigest()


def _referenced_plot_data_paths(
    summaries: object,
    *,
    base_path: Path,
) -> list[Path]:
    raw_paths: list[str] = []

    def visit(value: object, *, collecting: bool = False) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, collecting=collecting or key == "exp_results_npy_files")
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item, collecting=collecting)
        elif collecting and isinstance(value, str) and value.strip():
            raw_paths.append(value.strip())

    visit(summaries)
    resolved: list[Path] = []
    for raw in raw_paths:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = base_path / candidate
        try:
            candidate = candidate.resolve(strict=True)
            candidate.relative_to(base_path)
        except (OSError, ValueError):
            raise PlotAggregationError(
                "Plot data reference must resolve to a regular file inside the "
                "experiment folder"
            ) from None
        resolved.append(candidate)
    return sorted(set(resolved))


def _audit_input_hashes(
    base_path: Path,
    summaries: object,
) -> list[dict[str, object]]:
    candidates: set[Path] = set()
    for name in (
        "research_idea.md",
        "research_plan.json",
        "claim_evidence_graph.json",
        "experiment_registry.jsonl",
        "figure_spec.json",
        "figure_spec.md",
    ):
        path = base_path / name
        if path.exists() or path.is_symlink():
            candidates.add(path)
    for name in (
        "baseline_summary.json",
        "research_summary.json",
        "ablation_summary.json",
    ):
        candidates.update(base_path.glob(f"logs/*/{name}"))
    candidates.update(_quality_plot_guidance_paths(base_path))
    candidates.update(_referenced_plot_data_paths(summaries, base_path=base_path))
    if len(candidates) > 512:
        raise PlotAggregationError("Plot input set exceeds the auditable file limit")

    inputs: list[dict[str, object]] = []
    for path in sorted(candidates):
        maximum = 8 * 1024**3 if path.suffix.casefold() == ".npy" else 100 * 1024**2
        size, digest = _hash_bounded_regular_file(
            path,
            maximum=maximum,
            label="plot input",
        )
        inputs.append(
            {
                "path": path.relative_to(base_path).as_posix(),
                "bytes": size,
                "sha256": digest,
            }
        )
    return inputs


def _sanitized_isolation_receipt(isolation: dict[str, Any]) -> dict[str, Any]:
    receipt = dict(isolation)
    mounts = list(receipt.pop("read_only_mounts", []) or [])
    receipt["read_only_mount_count"] = len(mounts)
    return receipt


def _plot_attempt_receipt(
    *,
    index: int,
    code: str,
    result: PlotRunResult,
) -> dict[str, Any]:
    output = result.output.encode("utf-8", errors="replace")
    return {
        "attempt": index,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "output_bytes": len(output),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "succeeded": result.succeeded,
        "error_type": result.exc_type,
        "execution_backend": result.execution_backend,
        "isolation": _sanitized_isolation_receipt(result.isolation),
    }


def _require_plot_publication_lock_capability():
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - exercised via import guard
        raise PlotAggregationError(
            "Atomic plot publication requires a POSIX fcntl file-lock "
            "implementation; this platform is not supported"
        ) from exc
    return fcntl


@contextmanager
def _plot_publish_lock(base_path: Path):
    """Fail fast when another process is publishing this experiment's bundle."""

    fcntl = _require_plot_publication_lock_capability()

    try:
        canonical_base = base_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PlotAggregationError(
            "Could not resolve the experiment folder for plot publication"
        ) from exc
    lock_key = hashlib.sha256(str(canonical_base).encode("utf-8")).hexdigest()
    runtime_root = Path(tempfile.gettempdir()) / f"xscientist-plot-locks-{os.getuid()}"
    try:
        runtime_root.mkdir(mode=0o700, parents=False, exist_ok=True)
        root_path_metadata = runtime_root.lstat()
        root_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        root_flags |= getattr(os, "O_DIRECTORY", 0)
        root_flags |= getattr(os, "O_NOFOLLOW", 0)
        root_descriptor = os.open(runtime_root, root_flags)
        root_metadata = os.fstat(root_descriptor)
    except OSError as exc:
        raise PlotAggregationError("Could not prepare the plot lock directory") from exc
    if (
        stat.S_ISLNK(root_path_metadata.st_mode)
        or stat.S_ISLNK(root_metadata.st_mode)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or root_path_metadata.st_dev != root_metadata.st_dev
        or root_path_metadata.st_ino != root_metadata.st_ino
        or root_metadata.st_uid != os.getuid()
        or stat.S_IMODE(root_metadata.st_mode) & 0o077
    ):
        os.close(root_descriptor)
        raise PlotAggregationError("Plot lock directory has unsafe ownership or mode")
    lock_name = f"{lock_key}.lock"
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_name, flags, 0o600, dir_fd=root_descriptor)
    except OSError as exc:
        os.close(root_descriptor)
        raise PlotAggregationError(
            "Could not safely open the plot publish lock"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise PlotAggregationError("Plot publish lock is not a regular file")
        path_metadata = os.stat(
            lock_name,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if (
            stat.S_ISLNK(path_metadata.st_mode)
            or path_metadata.st_dev != metadata.st_dev
            or path_metadata.st_ino != metadata.st_ino
        ):
            raise PlotAggregationError("Plot publish lock identity changed")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PlotAggregationError(
                "Another plot publication is already active for this experiment"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)
        os.close(root_descriptor)


def _publish_plot_bundle_unlocked(
    *,
    base_path: Path,
    execution_path: Path,
    model: str,
    aggregator_code: str,
    plot_run: PlotRunResult,
    validated_outputs: list[dict[str, object]],
    execution_settings: PlotExecutionSettings,
    execution_config_sha256: str | None,
    attempt_count: int,
    attempt_receipts: list[dict[str, Any]],
    input_hashes: list[dict[str, object]],
) -> dict[str, Any]:
    """Transactionally replace the public plot bundle after full validation."""

    target_script = base_path / "auto_plot_aggregator.py"
    target_figures = base_path / "figures"
    target_receipt = base_path / "plot_execution_receipt.json"
    old_script = _optional_regular_payload(
        target_script,
        maximum=MAX_AGGREGATOR_CODE_BYTES,
        label="previous_plot_aggregator",
    )
    old_receipt = _optional_regular_payload(
        target_receipt,
        maximum=2 * 1024 * 1024,
        label="previous_plot_receipt",
    )

    target_figures_exists = False
    try:
        figures_metadata = target_figures.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise PlotAggregationError("Could not inspect existing figures") from exc
    else:
        if not stat.S_ISDIR(figures_metadata.st_mode):
            raise PlotAggregationError(
                "Refusing to replace a non-directory figures path"
            )
        target_figures_exists = True

    prepared_figures = Path(
        tempfile.mkdtemp(prefix=".xscientist-figures-publish-", dir=base_path.parent)
    )
    try:
        for output in validated_outputs:
            name = str(output["path"])
            if Path(name).name != name:
                raise PlotAggregationError("Generated figure has an unsafe filename")
            payload = read_bounded_regular_file(
                execution_path / "figures" / name,
                maximum=MAX_FIGURE_BYTES,
                label="plot_figure_publish",
            )
            if hashlib.sha256(payload).hexdigest() != output["sha256"]:
                raise PlotAggregationError(
                    f"Generated figure changed before publication: {name}"
                )
            atomic_write_bytes(prepared_figures / name, payload)

        policy_receipt = asdict(execution_settings.policy)
        policy_receipt["read_only_mount_count"] = len(
            policy_receipt.pop("read_only_mounts", []) or []
        )
        receipt: dict[str, Any] = {
            "schema": "xscientist.plot-execution-receipt.v1",
            "bundle_id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "script": {
                "path": "auto_plot_aggregator.py",
                "bytes": len(aggregator_code.encode("utf-8")),
                "sha256": hashlib.sha256(aggregator_code.encode("utf-8")).hexdigest(),
            },
            "execution_config_sha256": execution_config_sha256,
            "requested_execution": policy_receipt,
            "actual_execution": _sanitized_isolation_receipt(plot_run.isolation),
            "execution_backend": plot_run.execution_backend,
            "attempt_count": attempt_count,
            "evidence_root_read_only": execution_settings.policy.require_isolation,
            "inputs": input_hashes,
            "outputs": validated_outputs,
            "attempts": attempt_receipts,
            "selected_attempt": attempt_count,
            "status": "succeeded",
        }

        backup_figures = base_path.parent / (
            f".{base_path.name}.figures-backup-{uuid.uuid4().hex}"
        )
        old_moved = False
        new_installed = False
        try:
            if target_figures_exists:
                os.replace(target_figures, backup_figures)
                old_moved = True
            os.replace(prepared_figures, target_figures)
            new_installed = True
            atomic_write_bytes(target_script, aggregator_code.encode("utf-8"))
            atomic_write_json(target_receipt, receipt, indent=2, ensure_ascii=False)
        except Exception as exc:
            if new_installed:
                try:
                    shutil.rmtree(target_figures)
                except OSError:
                    pass
            if old_moved:
                try:
                    os.replace(backup_figures, target_figures)
                except OSError:
                    pass
            if old_script is None:
                target_script.unlink(missing_ok=True)
            else:
                atomic_write_bytes(target_script, old_script)
            if old_receipt is None:
                target_receipt.unlink(missing_ok=True)
            else:
                atomic_write_bytes(target_receipt, old_receipt)
            if isinstance(exc, PlotAggregationError):
                raise
            raise PlotAggregationError(
                "Verified plot bundle could not be published transactionally"
            ) from exc
        else:
            if old_moved:
                shutil.rmtree(backup_figures, ignore_errors=True)
        return receipt
    except BoundedFileError as exc:
        raise PlotAggregationError(
            f"Generated plot bundle changed before publication: {exc.reason}"
        ) from exc
    finally:
        if prepared_figures.exists():
            shutil.rmtree(prepared_figures, ignore_errors=True)


def _publish_plot_bundle(
    *,
    base_path: Path,
    execution_path: Path,
    model: str,
    aggregator_code: str,
    plot_run: PlotRunResult,
    validated_outputs: list[dict[str, object]],
    execution_settings: PlotExecutionSettings,
    execution_config_sha256: str | None,
    attempt_count: int,
    attempt_receipts: list[dict[str, Any]],
    input_hashes: list[dict[str, object]],
) -> dict[str, Any]:
    """Serialize old-bundle snapshot, replacement, and rollback as one lock scope."""

    with _plot_publish_lock(base_path):
        return _publish_plot_bundle_unlocked(
            base_path=base_path,
            execution_path=execution_path,
            model=model,
            aggregator_code=aggregator_code,
            plot_run=plot_run,
            validated_outputs=validated_outputs,
            execution_settings=execution_settings,
            execution_config_sha256=execution_config_sha256,
            attempt_count=attempt_count,
            attempt_receipts=attempt_receipts,
            input_hashes=input_hashes,
        )


def _generate_plot_bundle(
    base_folder: str,
    execution_folder: str | os.PathLike[str],
    model: str = "o1-2024-12-17",
    n_reflections: int = 5,
    *,
    execution_settings: PlotExecutionSettings,
) -> tuple[
    str,
    PlotRunResult,
    list[dict[str, object]],
    int,
    list[dict[str, Any]],
    list[dict[str, object]],
]:
    sandbox_policy = execution_settings.policy
    try:
        isolation_probe = Interpreter(
            working_dir=execution_folder,
            agent_file_name="auto_plot_aggregator.py",
            timeout=execution_settings.timeout,
            env_vars={},
            sandbox_policy=sandbox_policy,
        )
    except Exception as exc:
        raise PlotExecutionPolicyError(
            "Generated plotting code executor is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    isolation_probe.cleanup_session()

    filename = "auto_plot_aggregator.py"
    aggregator_script_path = os.path.join(execution_folder, filename)
    figures_dir = os.path.join(execution_folder, "figures")

    idea_text = load_idea_text(base_folder)
    exp_summaries = load_exp_summaries(base_folder)
    _link_referenced_plot_data(
        base_path=Path(base_folder).resolve(),
        execution_path=Path(execution_folder).resolve(),
        summaries=exp_summaries,
    )
    filtered_summaries_for_plot_agg = filter_experiment_summaries(
        exp_summaries, step_name="plot_aggregation"
    )
    claim_evidence_graph = load_contract_artifact(
        base_folder,
        "claim_evidence_graph",
        default={},
    )
    figure_spec = build_figure_spec_from_summaries(
        filtered_summaries_for_plot_agg,
        claim_evidence_graph=(
            claim_evidence_graph if isinstance(claim_evidence_graph, dict) else {}
        ),
        base_folder=base_folder,
        max_figures=MAX_FIGURES,
    )
    save_figure_spec(base_folder, figure_spec, producer="perform_plotting")
    figure_spec_markdown = os.path.join(base_folder, "figure_spec.md")
    with open(figure_spec_markdown, "w", encoding="utf-8") as f:
        f.write(render_figure_spec_markdown(figure_spec))
    input_hashes = _audit_input_hashes(Path(base_folder).resolve(), exp_summaries)
    # Convert them to one big JSON string for context
    combined_summaries_str = json.dumps(filtered_summaries_for_plot_agg, indent=2)
    quality_guidance_text = load_quality_plot_guidance(base_folder)

    # Build aggregator prompt
    aggregator_prompt = build_aggregator_prompt(
        combined_summaries_str,
        idea_text,
        quality_guidance_text,
        render_figure_spec_markdown(figure_spec),
    )

    # Call LLM
    client, model_name = create_client(model)
    response, msg_history = None, []
    try:
        response, msg_history = get_response_from_llm(
            prompt=aggregator_prompt,
            client=client,
            model=model_name,
            system_message=AGGREGATOR_SYSTEM_MSG,
            print_debug=False,
            msg_history=msg_history,
        )
    except Exception:
        traceback.print_exc()
        raise PlotAggregationError("Failed to obtain an aggregator script") from None

    aggregator_code = extract_code_snippet(response)
    if not aggregator_code.strip():
        raise PlotAggregationError(
            "The plotting model did not return executable Python code"
        )

    # First run of aggregator script
    plot_run = run_aggregator_script(
        aggregator_code,
        aggregator_script_path,
        execution_folder,
        filename,
        sandbox_policy=sandbox_policy,
        timeout=execution_settings.timeout,
    )
    attempt_count = 1
    attempt_receipts = [
        _plot_attempt_receipt(
            index=attempt_count, code=aggregator_code, result=plot_run
        )
    ]
    aggregator_out = plot_run.output

    # Multiple reflection loops
    for i in range(n_reflections):
        # Check number of figures
        figure_count = 0
        if os.path.exists(figures_dir):
            figure_count = len(
                [
                    f
                    for f in os.listdir(figures_dir)
                    if os.path.isfile(os.path.join(figures_dir, f))
                ]
            )
        print(f"[{i + 1} / {n_reflections}]: Number of figures: {figure_count}")
        # Reflection prompt with reminder for common checks and early exit
        reflection_prompt = f"""We have run your aggregator script and it produced {figure_count} figure(s). The script's output is:
```
{aggregator_out}
```

Please criticize the current script for any flaws including but not limited to:
- Are these enough plots for a final paper submission? Don't create more than {MAX_FIGURES} plots.
- Have you made sure to both use key numbers and generate more detailed plots from .npy files?
- Does the figure title and legend have informative and descriptive names? These plots are the final versions, ensure there are no comments or other notes.
- Can you aggregate multiple plots into one figure if suitable?
- Do the labels have underscores? If so, replace them with spaces.
- Make sure that every plot is unique and not duplicated from the original plots.
- Did you follow the visualization guidance file for what should be emphasized in captions, titles, architecture visuals, and experimental analysis?

If you believe you are done, simply say: "I am done". Otherwise, please provide an updated aggregator script in triple backticks."""

        print("[green]Reflection prompt:[/green] ", reflection_prompt)
        try:
            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=client,
                model=model_name,
                system_message=AGGREGATOR_SYSTEM_MSG,
                print_debug=False,
                msg_history=msg_history,
            )

        except Exception:
            traceback.print_exc()
            raise PlotAggregationError(
                "Failed to obtain a plotting-script reflection"
            ) from None

        # Early-exit check
        if figure_count > 0 and "I am done" in reflection_response:
            print("LLM indicated it is done with reflections. Exiting reflection loop.")
            break

        aggregator_new_code = extract_code_snippet(reflection_response)

        # If new code is provided and differs, run again
        if (
            aggregator_new_code.strip()
            and aggregator_new_code.strip() != aggregator_code.strip()
        ):
            aggregator_code = aggregator_new_code
            _clear_staged_figures(figures_dir)
            plot_run = run_aggregator_script(
                aggregator_code,
                aggregator_script_path,
                execution_folder,
                filename,
                sandbox_policy=sandbox_policy,
                timeout=execution_settings.timeout,
            )
            attempt_count += 1
            attempt_receipts.append(
                _plot_attempt_receipt(
                    index=attempt_count,
                    code=aggregator_code,
                    result=plot_run,
                )
            )
            aggregator_out = plot_run.output
        else:
            print(
                f"No new aggregator script was provided or it was identical. Reflection step {i+1} complete."
            )

    if not plot_run.succeeded:
        raise PlotAggregationError(
            "Generated plotting code still failed after all reflection attempts"
        )
    validated_outputs = _validate_plot_outputs(
        figures_dir=figures_dir,
        aggregator_script_path=aggregator_script_path,
        expected_code=aggregator_code,
    )
    if _audit_input_hashes(Path(base_folder).resolve(), exp_summaries) != input_hashes:
        raise PlotAggregationError("Plot inputs changed during isolated execution")
    return (
        aggregator_code,
        plot_run,
        validated_outputs,
        attempt_count,
        attempt_receipts,
        input_hashes,
    )


def aggregate_plots(
    base_folder: str,
    model: str = "o1-2024-12-17",
    n_reflections: int = 5,
    *,
    execution_config_path: str | os.PathLike[str] | None = None,
    allow_unisolated_local_model_code: bool = False,
) -> None:
    """Generate plots in a staging workspace, then publish a verified bundle."""

    base_path = Path(base_folder).expanduser().resolve()
    if not base_path.is_dir():
        raise PlotExecutionPolicyError("Experiment folder must be a directory")
    # Check this before creating a staging workspace, asking a model for code, or
    # invoking the isolated executor. There is deliberately no unlocked publish
    # fallback on platforms without the required file-lock primitive.
    _require_plot_publication_lock_capability()
    execution_settings = _plot_execution_settings(
        execution_config_path,
        allow_unisolated_local_model_code=allow_unisolated_local_model_code,
    )

    execution_config_sha256 = None
    if execution_config_path is not None:
        config_path = Path(execution_config_path).expanduser().resolve()
        try:
            config_payload = read_bounded_regular_file(
                config_path,
                maximum=2 * 1024 * 1024,
                label="plot_execution_config",
            )
        except BoundedFileError as exc:
            raise PlotExecutionPolicyError(
                f"Plot execution config failed safety validation: {exc.reason}"
            ) from exc
        execution_config_sha256 = hashlib.sha256(config_payload).hexdigest()

    if execution_settings.policy.require_isolation:
        mounts = tuple(execution_settings.policy.read_only_mounts)
        if str(base_path) not in mounts:
            mounts = (*mounts, str(base_path))
        execution_settings = replace(
            execution_settings,
            policy=replace(
                execution_settings.policy,
                network="none",
                read_only_mounts=mounts,
            ),
        )

    with tempfile.TemporaryDirectory(
        prefix=".xscientist-plot-stage-",
        dir=base_path.parent,
    ) as staging:
        execution_path = Path(staging)
        _link_staged_plot_inputs(base_path, execution_path)
        (
            aggregator_code,
            plot_run,
            validated_outputs,
            attempt_count,
            attempt_receipts,
            input_hashes,
        ) = _generate_plot_bundle(
            str(base_path),
            execution_path,
            model=model,
            n_reflections=n_reflections,
            execution_settings=execution_settings,
        )
        _publish_plot_bundle(
            base_path=base_path,
            execution_path=execution_path,
            model=model,
            aggregator_code=aggregator_code,
            plot_run=plot_run,
            validated_outputs=validated_outputs,
            execution_settings=execution_settings,
            execution_config_sha256=execution_config_sha256,
            attempt_count=attempt_count,
            attempt_receipts=attempt_receipts,
            input_hashes=input_hashes,
        )


def main():
    require_login("图表聚合(perform_plotting)")

    parser = argparse.ArgumentParser(
        description="Generate and execute a final plot aggregation script with LLM assistance."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to the experiment folder with summary JSON files.",
    )
    parser.add_argument(
        "--model",
        default="o1-2024-12-17",
        help="LLM model to use (default: o1-2024-12-17).",
    )
    parser.add_argument(
        "--reflections",
        type=int,
        default=5,
        help="Number of reflection steps to attempt (default: 5).",
    )
    parser.add_argument(
        "--execution-config",
        help=(
            "BFTS YAML whose Docker image and resource limits must protect "
            "model-generated plotting code."
        ),
    )
    parser.add_argument(
        "--allow-unisolated-local-model-code",
        action="store_true",
        help=(
            "UNSAFE local development only: permit model-generated plotting code "
            "to execute without Docker. Never use for autonomous or publication runs."
        ),
    )
    args = parser.parse_args()
    aggregate_plots(
        base_folder=args.folder,
        model=args.model,
        n_reflections=args.reflections,
        execution_config_path=args.execution_config,
        allow_unisolated_local_model_code=args.allow_unisolated_local_model_code,
    )


if __name__ == "__main__":
    main()
