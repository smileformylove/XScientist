"""Version-matched Docker executor inspection and cache management."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from ._version import __version__

EXECUTOR_SCHEMA = "xscientist.executor-status.v1"
DOCKER_INSTALL_URL = "https://docs.docker.com/get-started/get-docker/"
_RECIPE_LABEL = "org.xscientist.executor-recipe"
_SOURCE_DIGEST_LABEL = "org.xscientist.source-digest"
_SOURCE_ROOT_FILES = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "MANIFEST.in",
)
_SOURCE_ROOT_DIRS = ("xscientist", "ai_scientist", "compat", "third_party/licenses")
_MAX_SOURCE_CONTEXT_FILES = 4_096
_MAX_SOURCE_CONTEXT_BYTES = 64 * 1024 * 1024
_MAX_WORKSPACE_PARENT_SCAN = 16
_DOCKER_INFO_TIMEOUT_SECONDS = 5
_DOCKER_IMAGE_TIMEOUT_SECONDS = 10
_DOCKER_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_DOCKER_IDENTITY_FORMAT = '{"id":{{json .Id}},"labels":{{json .Config.Labels}}}'


class ExecutorManagerError(RuntimeError):
    """Raised when executor metadata or Docker operations are invalid."""


@dataclass(frozen=True)
class _ExecutorIdentity:
    """The host runtime identity that an executor image must reproduce."""

    install_source: str
    revision: str
    source_root: Path | None = None
    source_url: str | None = None
    source_digest: str | None = None


@dataclass(frozen=True)
class _SourceContextSnapshot:
    digest: str
    files: tuple[tuple[Path, bytes, bool], ...]


@dataclass(frozen=True)
class _ExecutorRecipe:
    text: str
    digest: str
    package_spec: str


def _source_checkout_root() -> Path | None:
    source_root = Path(__file__).resolve().parents[1]
    if (source_root / "pyproject.toml").is_file() and (
        source_root / "xscientist"
    ).is_dir():
        return source_root
    return None


def _source_file_allowed(relative: Path) -> bool:
    """Mirror the explicit setuptools package-data contract, not the repo."""

    parts = relative.parts
    if not parts:
        return False
    if parts[0] in {"xscientist", "compat"}:
        return relative.suffix == ".py"
    if parts[0] == "third_party":
        return parts[:2] == ("third_party", "licenses") and relative.suffix == ".txt"
    if parts[0] != "ai_scientist":
        return False
    if relative.suffix == ".py":
        return True
    package_relative = Path(*parts[1:])
    patterns = (
        "blank_icbinb_latex/*.tex",
        "blank_icml_latex/*.tex",
        "fewshot_examples/*.json",
        "fewshot_examples/*.txt",
        "ideas/*.json",
        "ideas/*.md",
        "protocol/SPEC.md",
        "protocol/conformance/*",
        "protocol/schemas/*.json",
        "resources/configs/*.yaml",
        "treesearch/utils/viz_templates/*.html",
        "treesearch/utils/viz_templates/*.js",
    )
    return any(package_relative.match(pattern) for pattern in patterns)


def _source_context_snapshot(source_root: Path) -> _SourceContextSnapshot:
    """Capture only package/build inputs that may be sent to Docker."""

    selected: list[Path] = []
    for relative in _SOURCE_ROOT_FILES:
        candidate = source_root / relative
        if candidate.is_symlink():
            raise ExecutorManagerError(
                f"source build input {relative} must not be a symlink"
            )
        if candidate.is_file():
            selected.append(candidate)
    if not (source_root / "pyproject.toml").is_file():
        raise ExecutorManagerError("source build input pyproject.toml is missing")

    for relative in _SOURCE_ROOT_DIRS:
        directory = source_root / relative
        if directory.is_symlink():
            raise ExecutorManagerError(
                f"source build input {relative} must not be a symlink"
            )
        if not directory.is_dir():
            if relative in {"xscientist", "ai_scientist", "compat"}:
                raise ExecutorManagerError(
                    f"source build input directory {relative} is missing"
                )
            continue
        for candidate in directory.rglob("*"):
            if candidate.is_symlink():
                raise ExecutorManagerError(
                    "source build inputs must not contain symlinks"
                )
            if candidate.is_file() and _source_file_allowed(
                candidate.relative_to(source_root)
            ):
                selected.append(candidate)

    unique = sorted(
        set(selected), key=lambda path: path.relative_to(source_root).as_posix()
    )
    if len(unique) > _MAX_SOURCE_CONTEXT_FILES:
        raise ExecutorManagerError("source build context contains too many files")

    digest = hashlib.sha256()
    files: list[tuple[Path, bytes, bool]] = []
    total_bytes = 0
    for candidate in unique:
        relative = candidate.relative_to(source_root)
        try:
            data = candidate.read_bytes()
            executable = bool(candidate.stat().st_mode & stat.S_IXUSR)
        except OSError as exc:
            raise ExecutorManagerError(
                f"cannot read source build input {relative.as_posix()}"
            ) from exc
        total_bytes += len(data)
        if total_bytes > _MAX_SOURCE_CONTEXT_BYTES:
            raise ExecutorManagerError("source build context is too large")
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        digest.update(b"\x01" if executable else b"\x00")
        files.append((relative, data, executable))
    return _SourceContextSnapshot(digest=digest.hexdigest(), files=tuple(files))


def _write_source_context(
    snapshot: _SourceContextSnapshot,
    destination: Path,
) -> None:
    for relative, data, executable in snapshot.files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o755 if executable else 0o644)


def _source_checkout_head(source_root: Path) -> str:
    """Return the current checkout HEAD or fail instead of inventing an identity."""

    try:
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ExecutorManagerError(
            "cannot determine the XScientist source checkout revision with git"
        ) from exc
    revision = revision_result.stdout.strip()
    if revision_result.returncode or not re.fullmatch(r"[0-9a-fA-F]{7,64}", revision):
        raise ExecutorManagerError(
            "cannot determine the XScientist source checkout revision with git"
        )
    return revision.lower()


def _source_checkout_state(source_root: Path) -> tuple[str, str]:
    """Return runtime revision plus the exact controlled-context digest."""

    snapshot = _source_context_snapshot(source_root)
    snapshot_revision = f"snapshot.{snapshot.digest[:16]}"
    try:
        revision = _source_checkout_head(source_root)
    except ExecutorManagerError:
        # GitHub source archives and unpacked editable installs have no Git
        # metadata.  Their bounded package snapshot is still an exact,
        # path-free build identity.
        return snapshot_revision, snapshot.digest
    try:
        dirty_result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *_SOURCE_ROOT_FILES,
                *_SOURCE_ROOT_DIRS,
            ],
            cwd=source_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return snapshot_revision, snapshot.digest
    if dirty_result.returncode:
        return snapshot_revision, snapshot.digest
    if dirty_result.stdout.strip():
        revision = f"{revision}-dirty.{snapshot.digest[:16]}"
    return revision, snapshot.digest


def _source_checkout_revision(source_root: Path) -> str:
    """Return Git state or a content-addressed source snapshot revision."""

    revision, _source_digest = _source_checkout_state(source_root)
    return revision


def _expected_executor_identity() -> _ExecutorIdentity:
    """Resolve the exact source identity expected inside the executor."""

    source_root = _source_checkout_root()
    if source_root is not None:
        revision, source_digest = _source_checkout_state(source_root)
        return _ExecutorIdentity(
            install_source="local-source",
            revision=revision,
            source_root=source_root,
            source_digest=source_digest,
        )

    # Keep the PEP 610 URL validation in onboarding, where the same source is
    # used to render a credential-free, immutable installation command.
    from .onboarding import _installed_runtime_source

    installed = _installed_runtime_source()
    if not installed.reproducible:
        raise ExecutorManagerError(installed.error)
    return _ExecutorIdentity(
        install_source=installed.install_source,
        revision=installed.revision,
        source_url=installed.source_url,
    )


def _capability_package_spec(dockerfile_text: str) -> str:
    """Extract the generated capability extras without trusting shell syntax."""

    matches = re.findall(
        r"(?:/tmp/xscientist-build-context|\bxscientist)"
        r"(\[[A-Za-z0-9._-]+(?:,[A-Za-z0-9._-]+)*\])?",
        dockerfile_text,
    )
    extras = {match for match in matches if match}
    if len(extras) > 1:
        raise ExecutorManagerError(
            "Dockerfile.executor contains conflicting capability recipes"
        )
    return "xscientist" + (next(iter(extras)) if extras else "")


def _normalized_recipe_text(dockerfile_text: str) -> str:
    normalized = dockerfile_text.replace("\r\n", "\n").replace("\r", "\n")
    for name in (
        "XSCIENTIST_VERSION",
        "XSCIENTIST_INSTALL_MODE",
        "XSCIENTIST_SOURCE_REVISION",
        "XSCIENTIST_SOURCE_DIGEST",
        "XSCIENTIST_INSTALL_SOURCE",
        "XSCIENTIST_RUNTIME_SPEC",
        "XSCIENTIST_RECIPE_DIGEST",
    ):
        normalized = re.sub(
            rf"(?m)^ARG {name}(?:=.*)?$",
            f"ARG {name}=<runtime>",
            normalized,
        )
    normalized = re.sub(
        r'(?m)^(\s*else \\\n\s*python -m pip install --no-cache-dir )"[^"\n]+"(; \\)$',
        r'\1"$XSCIENTIST_RUNTIME_SPEC"\2',
        normalized,
    )
    return normalized


def _executor_recipe_digest_from_text(dockerfile_text: str) -> str:
    package_spec = _capability_package_spec(dockerfile_text)
    digest = hashlib.sha256()
    digest.update(b"xscientist-executor-recipe-v1\x00")
    digest.update(package_spec.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(_normalized_recipe_text(dockerfile_text).encode("utf-8"))
    return digest.hexdigest()


def _executor_recipe(workspace: Path) -> _ExecutorRecipe:
    dockerfile = workspace / "Dockerfile.executor"
    if dockerfile.is_symlink() or not dockerfile.is_file():
        raise ExecutorManagerError("Dockerfile.executor was not found or is unsafe")
    try:
        text = dockerfile.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ExecutorManagerError("cannot read Dockerfile.executor") from exc
    return _ExecutorRecipe(
        text=text,
        digest=_executor_recipe_digest_from_text(text),
        package_spec=_capability_package_spec(text),
    )


def _runtime_spec(identity: _ExecutorIdentity, recipe: _ExecutorRecipe) -> str:
    if identity.install_source == "local-source":
        extras = recipe.package_spec.removeprefix("xscientist")
        return f"/tmp/xscientist-build-context{extras}"
    if identity.install_source == "vcs-commit":
        if not identity.source_url:
            raise ExecutorManagerError(
                "installed VCS runtime has no safe reproducible source URL"
            )
        return f"{recipe.package_spec} @ git+{identity.source_url}@{identity.revision}"
    if identity.install_source == "pypi-release":
        return f"{recipe.package_spec}=={__version__}"
    raise ExecutorManagerError("unsupported executor install source identity")


def _executor_build_arguments(
    identity: _ExecutorIdentity,
    recipe: _ExecutorRecipe,
) -> list[str]:
    values = {
        "XSCIENTIST_VERSION": __version__,
        "XSCIENTIST_INSTALL_MODE": (
            "local" if identity.install_source == "local-source" else "pypi"
        ),
        "XSCIENTIST_SOURCE_REVISION": identity.revision,
        "XSCIENTIST_SOURCE_DIGEST": identity.source_digest or "not-applicable",
        "XSCIENTIST_INSTALL_SOURCE": identity.install_source,
        "XSCIENTIST_RUNTIME_SPEC": _runtime_spec(identity, recipe),
        "XSCIENTIST_RECIPE_DIGEST": recipe.digest,
    }
    arguments: list[str] = []
    for name, value in values.items():
        arguments.extend(("--build-arg", f"{name}={value}"))
    return arguments


def _materialized_dockerfile(recipe: _ExecutorRecipe) -> str:
    """Upgrade generated legacy Dockerfiles in-memory to the current contract."""

    text = recipe.text.replace("\r\n", "\n").replace("\r", "\n")
    argument_names = (
        "XSCIENTIST_VERSION",
        "XSCIENTIST_INSTALL_MODE",
        "XSCIENTIST_SOURCE_REVISION",
        "XSCIENTIST_SOURCE_DIGEST",
        "XSCIENTIST_INSTALL_SOURCE",
        "XSCIENTIST_RUNTIME_SPEC",
        "XSCIENTIST_RECIPE_DIGEST",
    )
    missing = [
        name
        for name in argument_names
        if re.search(rf"(?m)^ARG {name}(?:=.*)?$", text) is None
    ]
    if missing:
        insertion = "".join(f"ARG {name}\n" for name in missing)
        text, count = re.subn(
            r"(?m)^(FROM [^\n]+\n)",
            rf"\1{insertion}",
            text,
            count=1,
        )
        if not count:
            raise ExecutorManagerError("Dockerfile.executor must contain a FROM line")

    if "$XSCIENTIST_RUNTIME_SPEC" not in text:
        text, replaced = re.subn(
            r'(?m)^(\s*else \\\n\s*python -m pip install --no-cache-dir )"[^"\n]+"(; \\)$',
            r'\1"$XSCIENTIST_RUNTIME_SPEC"\2',
            text,
            count=1,
        )
        if not replaced:
            text = (
                text.rstrip()
                + '\n\nRUN python -m pip install --no-cache-dir "$XSCIENTIST_RUNTIME_SPEC"\n'
            )
    text = text.rstrip() + f"""

LABEL org.opencontainers.image.version="$XSCIENTIST_VERSION" \\
      org.opencontainers.image.revision="$XSCIENTIST_SOURCE_REVISION" \\
      {_SOURCE_DIGEST_LABEL}="$XSCIENTIST_SOURCE_DIGEST" \\
      org.xscientist.install-source="$XSCIENTIST_INSTALL_SOURCE" \\
      {_RECIPE_LABEL}="$XSCIENTIST_RECIPE_DIGEST"
"""
    return text


def _workspace_config(workspace: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(workspace).expanduser().resolve()
    path = root / "bfts_config.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExecutorManagerError("bfts_config.yaml was not found") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ExecutorManagerError(
            f"cannot read bfts_config.yaml: {type(exc).__name__}"
        ) from exc
    if not isinstance(payload, dict):
        raise ExecutorManagerError("bfts_config.yaml must contain a mapping")
    return root, payload


def resolve_executor_workspace(start: str | Path) -> Path | None:
    """Find the nearest initialized executor workspace without an unbounded scan."""

    candidate = Path(start).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for depth, root in enumerate((candidate, *candidate.parents)):
        if depth >= _MAX_WORKSPACE_PARENT_SCAN:
            break
        has_config = (root / "bfts_config.yaml").is_file()
        has_recipe = (root / "Dockerfile.executor").is_file()
        initialized = (root / ".xscientist" / "providers.json").is_file()
        if (has_config and has_recipe) or initialized:
            return root
    return None


def _image_name(config: dict[str, Any]) -> str:
    exec_config = config.get("exec")
    image = str(
        exec_config.get("docker_image") if isinstance(exec_config, dict) else ""
    )
    if not image:
        image = f"xscientist-exec:{__version__}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]{0,254}", image):
        raise ExecutorManagerError("configured executor image name is invalid")
    return image


def inspect_executor(
    workspace: str | Path,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    root, config = _workspace_config(workspace)
    image = _image_name(config)
    expected_identity = _expected_executor_identity()
    expected_recipe = _executor_recipe(root)
    docker_available = shutil.which("docker") is not None
    labels: dict[str, str] = {}
    image_id: str | None = None
    daemon_ready = False
    image_available = False
    error = None
    if docker_available:
        try:
            info = run(
                ["docker", "info", "--format", "{{json .ServerVersion}}"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
                timeout=_DOCKER_INFO_TIMEOUT_SECONDS,
            )
            daemon_ready = info.returncode == 0
        except subprocess.TimeoutExpired:
            error = "Docker daemon identity check timed out"
        except OSError:
            error = "Docker daemon identity check could not start"
        if daemon_ready:
            try:
                inspected = run(
                    [
                        "docker",
                        "image",
                        "inspect",
                        image,
                        "--format",
                        _DOCKER_IDENTITY_FORMAT,
                    ],
                    cwd=root,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=_DOCKER_IMAGE_TIMEOUT_SECONDS,
                )
                image_available = inspected.returncode == 0
                if image_available:
                    try:
                        parsed = json.loads(inspected.stdout.strip() or "{}")
                        if not isinstance(parsed, dict):
                            raise ValueError("identity payload must be a mapping")
                        selected_id = str(parsed.get("id") or "").strip().lower()
                        selected_labels = parsed.get("labels")
                        if not _DOCKER_IMAGE_ID_PATTERN.fullmatch(selected_id):
                            raise ValueError("image ID is missing or invalid")
                        if selected_labels is None:
                            selected_labels = {}
                        if not isinstance(selected_labels, dict):
                            raise ValueError("image labels must be a mapping")
                        image_id = selected_id
                        labels = {
                            str(key): str(value)
                            for key, value in selected_labels.items()
                        }
                    except (json.JSONDecodeError, ValueError):
                        error = "executor image identity metadata is invalid"
            except subprocess.TimeoutExpired:
                error = "executor image identity check timed out"
            except OSError:
                error = "executor image identity check could not start"
    version = labels.get("org.opencontainers.image.version")
    source = labels.get("org.xscientist.install-source")
    revision = labels.get("org.opencontainers.image.revision")
    source_digest = labels.get(_SOURCE_DIGEST_LABEL)
    recipe_digest = labels.get(_RECIPE_LABEL)
    version_match = image_available and version == __version__
    install_source_match = (
        image_available and source == expected_identity.install_source
    )
    revision_match = image_available and revision == expected_identity.revision
    source_digest_match = image_available and (
        expected_identity.install_source != "local-source"
        or source_digest == expected_identity.source_digest
    )
    recipe_match = image_available and recipe_digest == expected_recipe.digest
    ready = (
        docker_available
        and daemon_ready
        and image_available
        and image_id is not None
        and version_match
        and install_source_match
        and revision_match
        and source_digest_match
        and recipe_match
    )
    if not docker_available:
        error = f"Docker CLI is not installed. Install Docker: {DOCKER_INSTALL_URL}"
    elif not daemon_ready and error is None:
        error = "Docker is installed, but its daemon is not running; start Docker"
    elif error is None and not image_available:
        error = f"executor image is not available locally: {image}"
    elif error is None and image_available and not version_match:
        error = f"executor version {version or 'unknown'} does not match {__version__}"
    elif error is None and image_available and not install_source_match:
        if source is None:
            error = "executor image is missing its install-source identity; rebuild it"
        else:
            error = (
                f"executor install source {source} does not match expected "
                f"{expected_identity.install_source}"
            )
    elif error is None and image_available and not revision_match:
        if revision is None:
            error = "executor image is missing its source revision; rebuild it"
        else:
            error = (
                f"executor source revision {revision} does not match expected "
                f"{expected_identity.revision}"
            )
    elif error is None and image_available and not source_digest_match:
        if source_digest is None:
            error = "executor image is missing its local source digest; rebuild it"
        else:
            error = (
                "executor local source content does not match the controlled snapshot"
            )
    elif error is None and image_available and not recipe_match:
        if recipe_digest is None:
            error = "executor image is missing its build recipe identity; rebuild it"
        else:
            error = (
                "executor build recipe does not match Dockerfile.executor capabilities"
            )
    if ready:
        next_action = None
    elif not docker_available:
        next_action = DOCKER_INSTALL_URL
    elif not daemon_ready:
        next_action = "Start Docker, then rerun this command"
    else:
        next_action = f"xscientist executor prepare --workspace {root.name}"
    return {
        "schema": EXECUTOR_SCHEMA,
        "ok": ready,
        "workspace": root.name,
        "image": image,
        "docker_available": docker_available,
        "daemon_ready": daemon_ready,
        "image_available": image_available,
        "image_id": image_id,
        "version": version,
        "expected_version": __version__,
        "version_match": version_match,
        "revision": revision,
        "expected_revision": expected_identity.revision,
        "revision_match": revision_match,
        "source_digest": source_digest,
        "expected_source_digest": expected_identity.source_digest,
        "source_digest_match": source_digest_match,
        "install_source": source,
        "expected_install_source": expected_identity.install_source,
        "install_source_match": install_source_match,
        "recipe_digest": recipe_digest,
        "expected_recipe_digest": expected_recipe.digest,
        "recipe_match": recipe_match,
        "error": error,
        "next_action": next_action,
        "host_paths_disclosed": False,
    }


def _source_build_arguments(_workspace: Path) -> tuple[Path | None, list[str]]:
    """Return source location plus explicit, reproducible Docker build arguments.

    The returned source root is an input to the controlled snapshotter; it is
    never passed directly to Docker as a build context.
    """

    identity = _expected_executor_identity()
    recipe = _executor_recipe(_workspace)
    return identity.source_root, _executor_build_arguments(identity, recipe)


def build_executor(
    workspace: str | Path,
    *,
    pull_base: bool = False,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    root, config = _workspace_config(workspace)
    if shutil.which("docker") is None:
        raise ExecutorManagerError(
            f"Docker CLI is not installed. Install Docker: {DOCKER_INSTALL_URL}"
        )
    image = _image_name(config)
    identity = _expected_executor_identity()
    recipe = _executor_recipe(root)
    source_args = _executor_build_arguments(identity, recipe)
    materialized_dockerfile = _materialized_dockerfile(recipe)
    source_snapshot = None
    if identity.source_root is not None:
        source_snapshot = _source_context_snapshot(identity.source_root)
        current_revision, current_digest = _source_checkout_state(identity.source_root)
        if (
            current_revision != identity.revision
            or current_digest != identity.source_digest
            or source_snapshot.digest != identity.source_digest
        ):
            raise ExecutorManagerError(
                "source checkout changed while preparing the executor; retry"
            )

    def run_build(*, build_context: Path, selected_dockerfile: Path) -> None:
        command = [
            "docker",
            "build",
            "-f",
            str(selected_dockerfile),
            "-t",
            image,
        ]
        if pull_base:
            command.append("--pull")
        command.extend(source_args)
        command.append(str(build_context))
        completed = run(command, cwd=build_context, text=True, check=False)
        if completed.returncode:
            raise ExecutorManagerError(
                f"executor build failed with return code {completed.returncode}"
            )

    # Docker always receives a controlled temporary context. Installed builds
    # contain only the selected Dockerfile; checkout builds add only the
    # package/build allowlist captured above, never the research workspace or
    # the complete source repository.
    with tempfile.TemporaryDirectory(prefix="xscientist-executor-build-") as raw:
        safe_context = Path(raw)
        if source_snapshot is not None:
            _write_source_context(source_snapshot, safe_context)
        safe_dockerfile = safe_context / "Dockerfile.executor"
        safe_dockerfile.write_text(materialized_dockerfile, encoding="utf-8")
        run_build(
            build_context=safe_context,
            selected_dockerfile=safe_dockerfile,
        )
    return inspect_executor(root, run=run)


def prepare_executor(
    workspace: str | Path,
    *,
    update: bool = False,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    status = inspect_executor(workspace, run=run)
    if status["ok"] and not update:
        return {**status, "cache_hit": True, "built": False}
    built = build_executor(workspace, pull_base=update, run=run)
    return {**built, "cache_hit": False, "built": True}


__all__ = [
    "DOCKER_INSTALL_URL",
    "EXECUTOR_SCHEMA",
    "ExecutorManagerError",
    "build_executor",
    "inspect_executor",
    "prepare_executor",
    "resolve_executor_workspace",
]
