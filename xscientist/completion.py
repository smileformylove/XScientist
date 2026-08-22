"""Dependency-free shell completion scripts for the public CLI."""

from __future__ import annotations

COMMANDS = (
    "explore demo start status audit history runs doctor setup auth provider capability executor research "
    "upgrade conformance benchmark metrics completion info init privacy serve help"
)

SUBCOMMANDS = {
    "runs": "list show watch logs cancel resume",
    "auth": "login status logout",
    "provider": "list check add activate remove test",
    "executor": "check build prepare update",
    "upgrade": "check",
    "conformance": "init check",
    "benchmark": "first-run autoresearch systems",
    "metrics": "status enable disable export",
    "capability": "list check",
    "privacy": "audit",
    "history": "list show diff save rollback",
    "research": (
        "guide start hypothesis plan preregister experiment evidence infer review "
        "claim status log dag"
    ),
}

OPTIONS = {
    "explore": (
        "--idea --expect --hypothesis --disprove --falsifier --test --success-rule "
        "--name --actor --lang --non-interactive --json"
    ),
    "demo": "--lang --open --autopilot --autopilot-profile --json",
    "start": (
        "--question --autopilot --task --provider --model --user --data-dir "
        "--allow-synthetic-data --max-project-tokens --max-project-hours "
        "--max-cost-usd --non-interactive --force --detach --prepare-only --json"
    ),
    "status": "--lang --verbose --json",
    "audit": "--ref --level --no-objects --json",
    "history": (
        "--limit --message --summary --actor --commit --from --to --deep --apply --json"
    ),
    "runs": "--workspace --json --interval --stream --tail --force",
    "doctor": "--workspace --task --provider --deep --json",
    "setup": "--task --provider --model --skip-credentials --deep --json",
    "provider": (
        "--workspace --model --max-cost-usd --live --timeout "
        "--non-interactive --json"
    ),
    "executor": "--workspace --json",
    "upgrade": "--workspace --online --timeout --json",
    "conformance": "--schema --json",
    "benchmark": "--tasks --workspace --profile --max-seconds --limit --kind --show-process --json",
    "metrics": "--json",
    "auth": "--user --lang --json",
    "init": "--profile --provider --model --force --json",
    "info": "--json",
    "serve": (
        "--host --port --work-dir --output-root --max-workers --state-dir "
        "--allow-unauthenticated"
    ),
}


def _bash_case(values: dict[str, str]) -> str:
    return "\n".join(
        f'    {name}) candidates="{candidates}" ;;'
        for name, candidates in values.items()
    )


def _zsh_case(values: dict[str, str], label: str) -> str:
    return "\n".join(
        f"    {name}) values=({candidates}); _describe '{label}' values ;;"
        for name, candidates in values.items()
    )


def completion_script(shell: str) -> str:
    """Return a useful completion script without modifying shell files."""

    if shell == "bash":
        return f"""# XScientist completion; generated locally.
_xscientist_complete() {{
  local current="${{COMP_WORDS[COMP_CWORD]}}"
  local command="${{COMP_WORDS[1]}}"
  local candidates=""
  if [[ $COMP_CWORD -eq 1 ]]; then
    candidates="{COMMANDS}"
  elif [[ $COMP_CWORD -eq 2 && $current != -* ]]; then
    case "$command" in
{_bash_case(SUBCOMMANDS)}
    esac
  fi
  if [[ -z $candidates && $current == -* ]]; then
    case "$command" in
{_bash_case(OPTIONS)}
    esac
  fi
  COMPREPLY=( $(compgen -W "$candidates" -- "$current") )
}}
complete -F _xscientist_complete xscientist
"""
    if shell == "zsh":
        return f"""#compdef xscientist
_xscientist() {{
  local command="${{words[2]}}"
  local -a values
  if (( CURRENT == 2 )); then
    values=({COMMANDS})
    _describe 'command' values
    return
  fi
  if (( CURRENT == 3 )) && [[ "${{words[CURRENT]}}" != -* ]]; then
    case "$command" in
{_zsh_case(SUBCOMMANDS, "subcommand")}
    esac
    return
  fi
  if [[ "${{words[CURRENT]}}" == -* ]]; then
    case "$command" in
{_zsh_case(OPTIONS, "option")}
    esac
  fi
}}
compdef _xscientist xscientist
"""
    if shell == "fish":
        lines = [
            "# XScientist completion; generated locally.",
            "complete -c xscientist -f",
        ]
        for command in COMMANDS.split():
            lines.append(
                "complete -c xscientist -n '__fish_use_subcommand' " f"-a {command}"
            )
        for command, candidates in SUBCOMMANDS.items():
            lines.append(
                "complete -c xscientist "
                f"-n '__fish_seen_subcommand_from {command}' "
                f"-a '{candidates}'"
            )
        for command, candidates in OPTIONS.items():
            for option in candidates.split():
                if option.startswith("--"):
                    lines.append(
                        "complete -c xscientist "
                        f"-n '__fish_seen_subcommand_from {command}' "
                        f"-l {option[2:]}"
                    )
        return "\n".join(lines) + "\n"
    raise ValueError(f"unsupported shell: {shell}")


__all__ = ["COMMANDS", "OPTIONS", "SUBCOMMANDS", "completion_script"]
