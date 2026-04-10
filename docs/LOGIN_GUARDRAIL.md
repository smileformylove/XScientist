# Login Guardrail

All user-facing operations now require a valid login session before execution.

## Why

- Prevent unauthenticated execution of project management and generation workflows.
- Enforce a single, consistent precondition: **login first, then operate**.

## Login Commands

Use the new CLI:

```bash
# Login
python3 auth_cli.py login --user <your_name>

# Check current status
python3 auth_cli.py status

# Logout
python3 auth_cli.py logout
```

Session file default path:

- `~/.ai_scientist/auth/session.json`

You can override via:

- `AI_SCIENTIST_AUTH_FILE=/custom/path/session.json`

## Enforced Entrypoints

The login guard is enforced in major operation entrypoints, including:

- `run_project.py`
- `research_manager.py`
- `continuous_paper_generator.py`
- `continuous_research_daemon.py`
- `run_daemon_profile.py`
- `run_daemon_rehearsal.py`
- `launch_scientist_bfts.py`
- `launch_scientist_zhipu.py`
- `preflight_check.py`
- `validate_repo.py`
- `ai_scientist/perform_ideation_temp_free.py`
- `ai_scientist/perform_writeup.py`
- `ai_scientist/perform_icbinb_writeup.py`
- `ai_scientist/perform_plotting.py`
- `start_research.sh`
- `run_stable_daemon.sh`

If not logged in, operations exit immediately with a clear remediation message.
