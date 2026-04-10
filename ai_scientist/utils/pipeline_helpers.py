from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Optional, Iterable


def _latex_commands(tex_stem: str = "template"):
    return [
        ["pdflatex", "-interaction=nonstopmode", f"{tex_stem}.tex"],
        ["bibtex", tex_stem],
        ["pdflatex", "-interaction=nonstopmode", f"{tex_stem}.tex"],
        ["pdflatex", "-interaction=nonstopmode", f"{tex_stem}.tex"],
    ]


def compile_latex(
    cwd: str | Path,
    pdf_file: str | Path | None = None,
    timeout: int = 30,
    tex_stem: str = "template",
    verbose: bool = True,
) -> bool:
    print("GENERATING LATEX")

    cwd_path = Path(cwd)
    compiled_pdf = cwd_path / f"{tex_stem}.pdf"

    for command in _latex_commands(tex_stem):
        try:
            result = subprocess.run(
                command,
                cwd=cwd_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            if verbose:
                print("Standard Output:\n", result.stdout)
                print("Standard Error:\n", result.stderr)
            if result.returncode != 0:
                print(
                    "EXCEPTION in compile_latex: "
                    f"command {' '.join(command)} exited with code {result.returncode}"
                )
                return False
        except FileNotFoundError:
            print(f"EXCEPTION in compile_latex: Missing executable {command[0]}")
            print(traceback.format_exc())
            return False
        except subprocess.TimeoutExpired:
            print(
                f"EXCEPTION in compile_latex: LaTeX timed out after {timeout} seconds."
            )
            print(traceback.format_exc())
            return False
        except subprocess.CalledProcessError:
            print(
                f"EXCEPTION in compile_latex: Error running command {' '.join(command)}"
            )
            print(traceback.format_exc())
            return False

    print("FINISHED GENERATING LATEX")

    if pdf_file is None:
        return compiled_pdf.exists()

    try:
        shutil.move(compiled_pdf, pdf_file)
        return True
    except FileNotFoundError:
        print("Failed to rename PDF.")
        print("EXCEPTION in compile_latex while moving PDF:")
        print(traceback.format_exc())
        return False


def get_available_gpus(gpu_ids: Optional[str] = None) -> list[int]:
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]

    try:
        import torch
    except ModuleNotFoundError as exc:
        print(f"Warning: torch unavailable while enumerating GPUs: {exc}")
        return []

    return list(range(torch.cuda.device_count()))


@contextmanager
def redirect_stdout_stderr_to_file(log_file_path: str | Path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_file_path, "a") as log:
        sys.stdout = log
        sys.stderr = log
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _pick_reflection_pdf(pdf_names: list[str]) -> Optional[str]:
    reflection_pdfs = [name for name in pdf_names if "reflection" in name.lower()]
    if not reflection_pdfs:
        return None

    final_pdfs = [name for name in reflection_pdfs if "final" in name.lower()]
    if final_pdfs:
        return sorted(final_pdfs)[-1]

    numbered_reflections = []
    for name in reflection_pdfs:
        match = re.search(r"reflection[_.]?(\d+)", name, re.IGNORECASE)
        if match:
            numbered_reflections.append((int(match.group(1)), name))

    if numbered_reflections:
        return max(numbered_reflections, key=lambda item: (item[0], item[1]))[1]

    return sorted(reflection_pdfs)[-1]


def find_best_pdf_path(
    output_dir: str | Path, prefer_reflections: bool = True
) -> Optional[str]:
    output_path = Path(output_dir)
    pdf_names = sorted(path.name for path in output_path.glob("*.pdf"))
    if not pdf_names:
        return None

    if prefer_reflections:
        preferred_name = _pick_reflection_pdf(pdf_names)
        if preferred_name is not None:
            return str(output_path / preferred_name)

    non_reflection = [name for name in pdf_names if "reflection" not in name.lower()]
    if non_reflection:
        return str(output_path / non_reflection[-1])

    return str(output_path / pdf_names[-1])


def find_latest_pdf_path(output_dir: str | Path) -> Optional[str]:
    return find_best_pdf_path(output_dir, prefer_reflections=False)


def _parse_bfts_run_index(run_name: str) -> Optional[int]:
    """
    Parse BFTS run directory names like '0-run' or '12-some_slug' into an integer index.

    Returns None for non-matching names.
    """
    match = re.match(r"^(\d+)-", run_name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def iter_bfts_run_dirs(
    base_folder: str | Path, *, logs_subdir: str = "logs", descending: bool = True
) -> Iterable[Path]:
    """
    Iterate BFTS run directories under '<base_folder>/<logs_subdir>/' ordered by run index.

    Example directory names: '0-run', '1-run', '2-some_slug'.
    """
    logs_dir = Path(base_folder) / logs_subdir
    if not logs_dir.exists():
        return []

    candidates: list[tuple[int, Path]] = []
    for entry in logs_dir.iterdir():
        if not entry.is_dir():
            continue
        run_idx = _parse_bfts_run_index(entry.name)
        if run_idx is None:
            continue
        candidates.append((run_idx, entry))

    candidates.sort(key=lambda item: item[0], reverse=descending)
    return [entry for _, entry in candidates]


def find_latest_bfts_run_dir(
    base_folder: str | Path, *, logs_subdir: str = "logs"
) -> Optional[Path]:
    """Return the newest (highest index) BFTS run dir, if any."""
    run_dirs = list(iter_bfts_run_dirs(base_folder, logs_subdir=logs_subdir, descending=True))
    return run_dirs[0] if run_dirs else None


def save_token_tracker(output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        from ai_scientist.utils.token_tracker import token_tracker

        summary = token_tracker.get_summary()
        interactions = token_tracker.get_interactions()
    except ModuleNotFoundError as exc:
        print(f"Warning: token tracker unavailable: {exc}")
        summary = {}
        interactions = {}

    with open(output_path / "token_tracker.json", "w") as f:
        json.dump(summary, f)

    with open(output_path / "token_tracker_interactions.json", "w") as f:
        json.dump(interactions, f)


def save_review_artifacts(
    output_dir: str | Path,
    text_review=None,
    image_review=None,
    *,
    text_filename: str = "review_text.json",
    image_filename: str = "review_img.json",
    text_mode: str = "json",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if text_review is not None:
        text_path = output_path / text_filename
        with open(text_path, "w", encoding="utf-8") as f:
            if text_mode == "json":
                json.dump(text_review, f, indent=4, ensure_ascii=False)
            elif text_mode == "text_json":
                f.write(json.dumps(text_review, indent=4, ensure_ascii=False))
            else:
                f.write(str(text_review))

    if image_review is not None:
        with open(output_path / image_filename, "w", encoding="utf-8") as f:
            json.dump(image_review, f, indent=4, ensure_ascii=False)


@lru_cache(maxsize=1)
def _process_cleanup_import_state():
    try:
        import psutil
        import signal
    except ModuleNotFoundError as exc:
        return None, None, exc
    return psutil, signal, None


def get_process_cleanup_capability() -> dict:
    psutil, _, exc = _process_cleanup_import_state()
    return {
        "available": psutil is not None,
        "backend": "psutil" if psutil is not None else "unavailable",
        "missing_module": getattr(exc, "name", None) if exc is not None else None,
        "reason": str(exc) if exc is not None else None,
    }


def cleanup_child_processes(
    keywords: Optional[list[str]] = None,
    timeout: int = 3,
    include_orphans: bool = False,
    workspace_root: str | Path | None = None,
    workspace_roots: Optional[list[str | Path]] = None,
    warn_if_unavailable: bool = True,
) -> dict:
    if keywords is None:
        keywords = ["python", "torch", "mp", "bfts", "experiment"]

    capability = get_process_cleanup_capability()
    if not capability["available"]:
        if warn_if_unavailable:
            print(f"Warning: process cleanup unavailable: {capability['reason']}")
        return {
            **capability,
            "children_found": 0,
            "children_killed": 0,
            "orphans_killed": 0,
        }

    psutil, signal, _ = _process_cleanup_import_state()

    children_killed = 0
    orphans_killed = 0

    def _terminate_process(process) -> bool:
        try:
            process.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

        try:
            process.wait(timeout=timeout)
        except psutil.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=timeout)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                return False

        return True

    def _safe_cmdline_parts(process) -> list[str]:
        try:
            return process.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []

    def _resolve_workspace_roots() -> list[Path]:
        resolved_roots: list[Path] = []
        for candidate in [workspace_root, *(workspace_roots or [])]:
            if candidate is None:
                continue
            try:
                root = Path(candidate).resolve()
            except OSError:
                continue
            if root not in resolved_roots:
                resolved_roots.append(root)
        return resolved_roots

    def _is_within_workspace(process, roots: list[Path]) -> bool:
        if not roots:
            return True

        cwd = process.info.get("cwd")
        if cwd:
            try:
                cwd_path = Path(cwd).resolve()
            except OSError:
                cwd_path = None
            if cwd_path is not None:
                for root in roots:
                    try:
                        cwd_path.relative_to(root)
                        return True
                    except ValueError:
                        continue

        cmdline_parts = _safe_cmdline_parts(process)
        for part in cmdline_parts:
            lowered_part = part.lower()
            for root in roots:
                root_str = str(root).lower()
                if root_str in lowered_part:
                    return True
                try:
                    part_path = Path(part).expanduser().resolve()
                except (OSError, RuntimeError):
                    continue
                try:
                    part_path.relative_to(root)
                    return True
                except ValueError:
                    continue
        return False

    protected_pids = {os.getpid()}

    current_process = psutil.Process()
    try:
        protected_pids.add(current_process.pid)
        protected_pids.update(parent.pid for parent in current_process.parents())
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, OSError):
        pass

    try:
        children = [
            child
            for child in current_process.children(recursive=True)
            if child.pid not in protected_pids
        ]
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, OSError):
        children = []

    for child in children:
        if _terminate_process(child):
            children_killed += 1

    if include_orphans:
        child_pids = {child.pid for child in children}
        roots = _resolve_workspace_roots()

        try:
            process_iter = psutil.process_iter(["pid", "ppid", "cmdline", "cwd"])
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, OSError):
            process_iter = []

        for proc in process_iter:
            try:
                if proc.pid in protected_pids or proc.pid in child_pids:
                    continue

                if proc.ppid() not in {0, 1}:
                    continue

                cmdline = " ".join(_safe_cmdline_parts(proc)).lower()
                if not cmdline or not any(keyword in cmdline for keyword in keywords):
                    continue

                if not _is_within_workspace(proc, roots):
                    continue

                if _terminate_process(proc):
                    orphans_killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, OSError):
                continue

    return {
        **capability,
        "children_found": len(children),
        "children_killed": children_killed,
        "orphans_killed": orphans_killed,
    }
