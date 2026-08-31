"""Dependency-free shell completion scripts for the public CLI."""

from __future__ import annotations

COMMANDS = (
    "project batch daemon manager ara auth feedback validate bfts zhipu preflight "
    "research evolution git serve info explore demo status audit history runs "
    "executor upgrade completion conformance benchmark metrics init start setup "
    "doctor capability provider privacy evolution-gate help"
)

RESEARCH_COMMANDS = (
    "doctor start guide hypothesis plan discovery program opportunity rollout "
    "rollout-audit verifier-authority belief belief-audit literature preregister confirm "
    "trajectory-bind attempt-disposition "
    "experiment evidence estimand effect infer ingest review claim init status audit "
    "context decide tree dag adapter fsck checkpoint record objects stage add unstage "
    "commit branch switch restore revert tag blame merge log trajectory show diff object bundle "
    "export reproduce"
)

RESEARCH_SUBCOMMANDS = {
    "discovery": "template plan assess",
    "program": (
        "template portfolio prediction prioritize posterior mechanism quality "
        "boundary review followup claim"
    ),
    "opportunity": "direction pool attempt judge grade allocate inspect",
    "verifier-authority": "prepare finalize verify",
    "literature": "plan receipt source update passage",
    "adapter": "list doctor sync",
    "object": "add",
    "bundle": "create verify restore",
}
RESEARCH_NESTED_PARSERS = tuple(
    command for command in RESEARCH_SUBCOMMANDS if command != "bundle"
)

SUBCOMMANDS = {
    "runs": "list show watch logs cancel resume",
    "auth": "login status logout",
    "provider": "list check add activate remove test",
    "executor": "check build prepare update",
    "upgrade": "check",
    "conformance": "init check",
    "benchmark": "first-run autoresearch systems verify",
    "metrics": "status enable disable export",
    "capability": "list check",
    "privacy": "audit",
    "history": "list show diff save rollback",
    "evolution": "candidate benchmark canary harness-audit attest deploy rollback",
    "research": RESEARCH_COMMANDS,
    "git": RESEARCH_COMMANDS,
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


def _subparser_choices(parser: object) -> dict[str, object]:
    for action in getattr(parser, "_actions", ()):
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            return dict(choices)
    return {}


def _parser_options(parser: object) -> str:
    options = {
        option
        for action in getattr(parser, "_actions", ())
        for option in getattr(action, "option_strings", ())
    }
    return " ".join(sorted(options, key=lambda item: (not item.startswith("--"), item)))


def _parser_option_tables(prefix: str, parser: object) -> dict[str, str]:
    """Return direct options for every exact argparse path below ``prefix``."""

    result = {prefix: _parser_options(parser)}
    for command, nested_parser in _subparser_choices(parser).items():
        result.update(_parser_option_tables(f"{prefix}::{command}", nested_parser))
    return result


def _root_option_tables() -> dict[str, str]:
    """Derive root-command options from the parsers that actually accept them."""

    from .cli import (
        _DELEGATES,
        _SPECIAL_COMMAND_OPTIONS,
        _build_capability_parser,
        _build_doctor_parser,
        _build_parser,
        _build_setup_parser,
        _build_start_parser,
    )

    # Delegate entries in the public parser are REMAINDER stubs, not option
    # contracts. Never treat those stubs as authoritative completion sources.
    parsers = {
        command: parser
        for command, parser in _subparser_choices(_build_parser()).items()
        if command not in _DELEGATES
    }
    parsers.update(
        {
            "start": _build_start_parser(),
            "setup": _build_setup_parser(),
            "doctor": _build_doctor_parser(),
            "capability": _build_capability_parser(),
        }
    )
    parsers.update(_delegate_parsers(set(_DELEGATES)))
    result: dict[str, str] = {}
    for command, parser in parsers.items():
        result.update(_parser_option_tables(command, parser))
    result.update(
        {
            command: " ".join(options)
            for command, options in _SPECIAL_COMMAND_OPTIONS.items()
        }
    )
    return result


def _delegate_parsers(delegates: set[str]) -> dict[str, object]:
    """Build lightweight real parsers for delegated root commands."""

    from ai_scientist.apps.auth import build_parser as build_auth_parser
    from ai_scientist.apps.batch_cli import build_parser as build_batch_parser
    from ai_scientist.apps.project_cli import build_parser as build_project_parser
    from ai_scientist.writing_prompt_profiles import (
        DEFAULT_WRITING_PROFILE,
        list_writing_profiles,
    )
    from ai_scientist.utils.workflow_modes import list_workflow_modes
    from xscientist.evolution_cli import _build_parser as build_evolution_parser

    common = {
        "default_writing_profile": DEFAULT_WRITING_PROFILE,
        "writing_profiles": list_writing_profiles(),
        "workflow_modes": list_workflow_modes(),
    }
    available = {
        "auth": build_auth_parser(),
        "batch": build_batch_parser(default_research_dir=".", **common),
        "evolution": build_evolution_parser(),
        "project": build_project_parser(default_output_root=".", **common),
    }
    return {
        command: parser for command, parser in available.items() if command in delegates
    }


def _research_option_tables() -> dict[str, str]:
    """Derive exact option candidates from the real nested research parser."""

    from .research_cli import _build_parser

    result: dict[str, str] = {}
    for command, parser in _subparser_choices(_build_parser()).items():
        result[f"research::{command}"] = _parser_options(parser)
        if command not in RESEARCH_NESTED_PARSERS:
            continue
        for nested, nested_parser in _subparser_choices(parser).items():
            result[f"research::{command}::{nested}"] = _parser_options(nested_parser)
    return result


# Public for compatibility, but generated from argparse rather than maintained as
# a second command schema that can silently drift from the executable CLI.
OPTIONS = _root_option_tables()


def _nested_option_path_selection(shell: str) -> str:
    """Select the deepest exact non-research argparse path in shell argv."""

    lines: list[str] = []
    paths = sorted(
        (key.split("::") for key in OPTIONS if "::" in key),
        key=lambda parts: (len(parts), parts),
    )
    for parts in paths:
        key = "::".join(parts)
        if shell == "bash":
            checks = [f"$command == {parts[0]}"]
            checks.extend(
                f'${{COMP_WORDS[{index + 1}]}} == "{part}"'
                for index, part in enumerate(parts[1:], start=1)
            )
            lines.append(
                f"    if [[ $COMP_CWORD -ge {len(parts) + 1} && "
                + " && ".join(checks)
                + f' ]]; then option_command="{key}"; fi'
            )
        elif shell == "zsh":
            checks = [f'"$command" == "{parts[0]}"']
            checks.extend(
                f'"${{words[{index + 2}]}}" == "{part}"'
                for index, part in enumerate(parts[1:], start=1)
            )
            lines.append(
                f"    if (( CURRENT >= {len(parts) + 2} )) && [[ "
                + " && ".join(checks)
                + f' ]]; then option_command="{key}"; fi'
            )
        else:  # pragma: no cover - internal misuse
            raise ValueError(f"unsupported option-path shell: {shell}")
    return "\n".join(lines)


def _fish_option_condition(key: str) -> str:
    parts = key.split("::")
    conditions = [
        f"__xscientist_word_is {index + 2} {part}" for index, part in enumerate(parts)
    ]
    immediate_children = sorted(
        candidate.split("::")[-1]
        for candidate in OPTIONS
        if candidate.startswith(f"{key}::")
        and candidate.count("::") == key.count("::") + 1
    )
    conditions.extend(
        f"not __xscientist_word_is {len(parts) + 2} {child}"
        for child in immediate_children
    )
    return "; and ".join(conditions)


def completion_script(shell: str) -> str:
    """Return a useful completion script without modifying shell files."""

    research_option_tables = _research_option_tables()
    option_tables = {**OPTIONS, **research_option_tables}
    nested_parser_pattern = "|".join(RESEARCH_NESTED_PARSERS)
    research_subcommand_pattern = "|".join(RESEARCH_SUBCOMMANDS)
    subcommand_pattern = "|".join(SUBCOMMANDS)

    if shell == "bash":
        return f"""# XScientist completion; generated locally.
_xscientist_complete() {{
  local current="${{COMP_WORDS[COMP_CWORD]}}"
  local command="${{COMP_WORDS[1]}}"
  local research_command="${{COMP_WORDS[2]}}"
  local option_command="$command"
  local candidates=""
  if [[ $COMP_CWORD -eq 1 ]]; then
    candidates="{COMMANDS}"
  elif [[ $COMP_CWORD -eq 2 && $current != -* ]]; then
    case "$command" in
      {subcommand_pattern})
        case "$command" in
{_bash_case(SUBCOMMANDS)}
        esac
        ;;
    esac
  elif [[ ( $command == research || $command == git ) && $COMP_CWORD -eq 3 && $current != -* ]]; then
    case "$research_command" in
      {research_subcommand_pattern})
        case "$research_command" in
{_bash_case(RESEARCH_SUBCOMMANDS)}
        esac
        ;;
    esac
  fi
  if [[ -z $candidates && $current == -* ]]; then
{_nested_option_path_selection("bash")}
    if [[ $command == research || $command == git ]]; then
      option_command="research::$research_command"
      case "$research_command" in
        {nested_parser_pattern})
          if [[ $COMP_CWORD -ge 4 && ${{COMP_WORDS[3]}} != -* ]]; then
            option_command="research::$research_command::${{COMP_WORDS[3]}}"
          fi
          ;;
      esac
    fi
    case "$option_command" in
{_bash_case(option_tables)}
    esac
  fi
  COMPREPLY=( $(compgen -W "$candidates" -- "$current") )
}}
complete -F _xscientist_complete -o bashdefault -o default xscientist
"""
    if shell == "zsh":
        return f"""#compdef xscientist
_xscientist() {{
  local command="${{words[2]}}"
  local research_command="${{words[3]}}"
  local option_command="$command"
  local -a values
  if (( CURRENT == 2 )); then
    values=({COMMANDS})
    _describe 'command' values
    return
  fi
  if (( CURRENT == 3 )) && [[ "${{words[CURRENT]}}" != -* ]]; then
    case "$command" in
      {subcommand_pattern})
        case "$command" in
{_zsh_case(SUBCOMMANDS, "subcommand")}
        esac
        return
        ;;
    esac
  fi
  if (( CURRENT == 4 )) && [[ "$command" == research || "$command" == git ]] && [[ "${{words[CURRENT]}}" != -* ]]; then
    case "$research_command" in
      {research_subcommand_pattern})
        case "$research_command" in
{_zsh_case(RESEARCH_SUBCOMMANDS, "research subcommand")}
        esac
        return
        ;;
    esac
  fi
  if [[ "${{words[CURRENT]}}" == -* ]]; then
{_nested_option_path_selection("zsh")}
    if [[ "$command" == research || "$command" == git ]]; then
      option_command="research::$research_command"
      case "$research_command" in
        {nested_parser_pattern})
          if (( CURRENT >= 5 )) && [[ "${{words[4]}}" != -* ]]; then
            option_command="research::$research_command::${{words[4]}}"
          fi
          ;;
      esac
    fi
    case "$option_command" in
{_zsh_case(option_tables, "option")}
    esac
    return
  fi
  _files
}}
compdef _xscientist xscientist
"""
    if shell == "fish":
        lines = [
            "# XScientist completion; generated locally.",
            "complete -c xscientist -f",
            "function __xscientist_word_is",
            "    set -l index $argv[1]",
            "    set -l expected $argv[2]",
            "    set -l words (commandline -opc)",
            '    test (count $words) -ge $index; and test "$words[$index]" = "$expected"',
            "end",
            "function __xscientist_depth_is",
            "    set -l words (commandline -opc)",
            "    test (count $words) -eq $argv[1]",
            "end",
            "function __xscientist_research_frontend",
            "    __xscientist_word_is 2 research; or __xscientist_word_is 2 git",
            "end",
        ]
        for command in COMMANDS.split():
            lines.append(
                "complete -c xscientist -n '__fish_use_subcommand' " f"-a {command}"
            )
        for command, candidates in SUBCOMMANDS.items():
            condition = (
                f"__xscientist_word_is 2 {command}; and " "__xscientist_depth_is 2"
            )
            lines.append(
                "complete -c xscientist " f"-n '{condition}' " f"-a '{candidates}'"
            )
        for command, candidates in RESEARCH_SUBCOMMANDS.items():
            lines.append(
                "complete -c xscientist "
                "-n '__xscientist_research_frontend; and "
                f"__xscientist_word_is 3 {command}; and "
                "__xscientist_depth_is 3' "
                f"-a '{candidates}'"
            )
        for command, candidates in OPTIONS.items():
            condition = _fish_option_condition(command)
            for option in candidates.split():
                if option.startswith("--"):
                    lines.append(
                        "complete -c xscientist "
                        f"-n '{condition}' "
                        f"-l {option[2:]}"
                    )
        for key, candidates in research_option_tables.items():
            _prefix, command, *nested = key.split("::")
            condition = (
                "__xscientist_research_frontend; and "
                f"__xscientist_word_is 3 {command}"
            )
            if nested:
                condition += f"; and __xscientist_word_is 4 {nested[0]}"
            elif command in RESEARCH_NESTED_PARSERS:
                condition += "".join(
                    f"; and not __xscientist_word_is 4 {candidate}"
                    for candidate in RESEARCH_SUBCOMMANDS[command].split()
                )
            for option in candidates.split():
                if option.startswith("--"):
                    lines.append(
                        "complete -c xscientist "
                        f"-n '{condition}' "
                        f"-l {option[2:]}"
                    )
        return "\n".join(lines) + "\n"
    raise ValueError(f"unsupported shell: {shell}")


__all__ = [
    "COMMANDS",
    "OPTIONS",
    "RESEARCH_COMMANDS",
    "RESEARCH_SUBCOMMANDS",
    "SUBCOMMANDS",
    "completion_script",
]
