import os
import os.path as osp
import json
import uuid
from pathlib import Path

import yaml

from ai_scientist.resources import resolve_bfts_config_path

from .utils.serialize import atomic_write_text


def idea_to_markdown(
    data: dict,
    output_path: str,
    load_code: str | None,
    *,
    research_plan: dict | None = None,
) -> None:
    """
    Convert a dictionary into a markdown file.

    Args:
        data: Dictionary containing the data to convert
        output_path: Path where the markdown file will be saved
        load_code: Path to a code file to include in the markdown
    """
    lines = []
    for key, value in data.items():
        header = key.replace("_", " ").title()
        lines.append(f"## {header}\n\n")
        if isinstance(value, (list, tuple)):
            lines.extend(f"- {item}\n" for item in value)
            lines.append("\n")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                lines.append(f"### {sub_key}\n")
                lines.append(f"{sub_value}\n\n")
        else:
            lines.append(f"{value}\n\n")

    if load_code:
        assert os.path.exists(
            load_code
        ), f"Code path at {load_code} must exist if using the 'load_code' flag. This is an optional code prompt that you may choose to include; if not, please do not set 'load_code'."
        with open(load_code, "r", encoding="utf-8") as code_file:
            code = code_file.read()
        lines.extend(
            [
                "## Code To Potentially Use\n\n",
                "Use the following code as context for your experiments:\n\n",
                f"```python\n{code}\n```\n\n",
            ]
        )

    if research_plan:
        contract = {
            "plan_id": research_plan.get("plan_id"),
            "workflow_mode": research_plan.get("workflow_mode"),
            "tasks": [
                item
                for item in research_plan.get("tasks") or []
                if isinstance(item, dict)
            ],
            "acceptance_rules": research_plan.get("acceptance_rules") or [],
            "required_discriminating_tests": research_plan.get(
                "required_discriminating_tests"
            )
            or [],
            "produced_artifacts": research_plan.get("produced_artifacts") or [],
            "execution_policy": research_plan.get("execution_policy") or {},
        }
        lines.extend(
            [
                "## Binding Research Contract\n\n",
                "The following plan is an execution constraint, not optional context. "
                "Run its required tasks and discriminating tests, preserve failed or "
                "refuting outcomes, and explicitly report every deviation. Do not claim "
                "completion when an acceptance rule or required artifact is missing.\n\n",
                "```json\n",
                json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True),
                "\n```\n\n",
            ]
        )

    atomic_write_text(output_path, "".join(lines))


def edit_bfts_config_file(
    config_path: str,
    idea_dir: str,
    idea_path: str,
    *,
    resume_from: str | None = None,
) -> str:
    """
    Edit the bfts_config.yaml file to point to the idea.md file

    Args:
        config_path: Path to the bfts_config.yaml file
        idea_dir: Directory where the idea.md file is located
        idea_path: Path to the idea.md file

    Returns:
        Path to a unique run config retained with the idea for provenance
    """
    config_dir = osp.join(idea_dir, ".xscientist", "configs")
    os.makedirs(config_dir, exist_ok=True)
    run_config_path = osp.join(
        config_dir, f"bfts_config-{os.getpid()}-{uuid.uuid4().hex}.yaml"
    )
    resolved_config_path = resolve_bfts_config_path(config_path)
    with open(resolved_config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config_base = Path(config_dir).resolve()

    def portable_reference(value: str | os.PathLike[str]) -> str:
        return Path(
            os.path.relpath(Path(value).expanduser().resolve(), config_base)
        ).as_posix()

    config["desc_file"] = portable_reference(idea_path)
    config["workspace_dir"] = portable_reference(idea_dir)

    # Autopilot may bind an explicitly hashed external dataset.  The executor
    # consumes it read-only; otherwise retain the legacy empty local fixture.
    data_dir = str(os.environ.get("AI_SCIENTIST_PROJECT_DATA_DIR") or "").strip()
    if not data_dir:
        data_dir = osp.join(idea_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
    config["data_dir"] = portable_reference(data_dir)

    # make an empty log directory
    log_dir = osp.join(idea_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    config["log_dir"] = portable_reference(log_dir)
    config["resume_from"] = portable_reference(resume_from) if resume_from else None

    atomic_write_text(run_config_path, yaml.dump(config))
    return run_config_path
