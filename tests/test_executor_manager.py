from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

import xscientist.executor_manager as executor_manager
from xscientist._version import __version__
from xscientist.cli import main as cli_main
from xscientist.executor_manager import (
    DOCKER_INSTALL_URL,
    build_executor,
    inspect_executor,
    prepare_executor,
)


class ExecutorManagerTests(unittest.TestCase):
    _IMAGE_ID = "sha256:" + "a" * 64

    def _identity_payload(self, labels: dict[str, str]) -> str:
        return json.dumps({"id": self._IMAGE_ID, "labels": labels})

    def _workspace(self, root: str) -> Path:
        workspace = Path(root) / "study"
        workspace.mkdir()
        (workspace / "bfts_config.yaml").write_text(
            yaml.safe_dump(
                {"exec": {"docker_image": f"xscientist-exec:{__version__}"}}
            ),
            encoding="utf-8",
        )
        (workspace / "Dockerfile.executor").write_text(
            """FROM python:3.11-slim
ARG XSCIENTIST_VERSION=0.1.4
ARG XSCIENTIST_INSTALL_MODE=pypi
ARG XSCIENTIST_SOURCE_REVISION=release
ARG XSCIENTIST_INSTALL_SOURCE=pypi-release
COPY . /tmp/xscientist-build-context
RUN if [ "$XSCIENTIST_INSTALL_MODE" = "local" ]; then \\
      python -m pip install --no-cache-dir "/tmp/xscientist-build-context[research,zhipu]"; \\
    else \\
      python -m pip install --no-cache-dir "xscientist[research,zhipu]==0.1.4"; \\
    fi
LABEL org.opencontainers.image.version="$XSCIENTIST_VERSION" \\
      org.opencontainers.image.revision="$XSCIENTIST_SOURCE_REVISION" \\
      org.xscientist.install-source="$XSCIENTIST_INSTALL_SOURCE"
""",
            encoding="utf-8",
        )
        return workspace

    def _recipe_digest(self, workspace: Path) -> str:
        return executor_manager._executor_recipe(workspace).digest

    def test_installed_build_uses_dockerfile_only_temporary_context(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = self._workspace(td)
            sentinel = workspace / "private-research-sentinel.txt"
            sentinel.write_text("must never reach the Docker daemon", encoding="utf-8")
            installed_module = (
                Path(td) / "site-packages" / "xscientist" / "executor_manager.py"
            )
            observed: dict[str, object] = {}

            def run(command, **kwargs):
                build_context = Path(command[-1])
                selected_dockerfile = Path(command[command.index("-f") + 1])
                observed.update(
                    {
                        "command": list(command),
                        "cwd": Path(kwargs["cwd"]),
                        "context": build_context,
                        "dockerfile": selected_dockerfile,
                        "dockerfile_text": selected_dockerfile.read_text(
                            encoding="utf-8"
                        ),
                        "context_files": sorted(
                            path.name for path in build_context.iterdir()
                        ),
                        "sentinel_present": (build_context / sentinel.name).exists(),
                    }
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            ready = {"ok": True, "image": f"xscientist-exec:{__version__}"}
            with (
                mock.patch(
                    "xscientist.executor_manager.shutil.which",
                    return_value="/usr/bin/docker",
                ),
                mock.patch(
                    "xscientist.executor_manager.__file__",
                    str(installed_module),
                ),
                mock.patch(
                    "xscientist.executor_manager._expected_executor_identity",
                    return_value=executor_manager._ExecutorIdentity(
                        install_source="vcs-commit",
                        revision="f" * 40,
                        source_url="https://example.invalid/XScientist.git",
                    ),
                ),
                mock.patch(
                    "xscientist.executor_manager.inspect_executor",
                    return_value=ready,
                ),
            ):
                payload = build_executor(workspace, pull_base=True, run=run)

        command = observed["command"]
        context = observed["context"]
        dockerfile = observed["dockerfile"]
        self.assertEqual(payload, ready)
        self.assertNotEqual(context, workspace)
        self.assertEqual(observed["cwd"], context)
        self.assertEqual(Path(dockerfile).parent, context)
        self.assertEqual(observed["context_files"], ["Dockerfile.executor"])
        self.assertIn(
            'python -m pip install --no-cache-dir "$XSCIENTIST_RUNTIME_SPEC"',
            observed["dockerfile_text"],
        )
        self.assertNotIn(
            "xscientist[research,zhipu]==0.1.4", observed["dockerfile_text"]
        )
        self.assertIn("org.xscientist.executor-recipe", observed["dockerfile_text"])
        self.assertIn("org.xscientist.source-digest", observed["dockerfile_text"])
        self.assertFalse(observed["sentinel_present"])
        self.assertIn("--pull", command)
        self.assertNotIn("XSCIENTIST_INSTALL_MODE=local", command)
        self.assertIn(f"XSCIENTIST_VERSION={__version__}", command)
        self.assertIn(f"XSCIENTIST_SOURCE_REVISION={'f' * 40}", command)
        self.assertIn("XSCIENTIST_INSTALL_SOURCE=vcs-commit", command)
        self.assertIn("XSCIENTIST_SOURCE_DIGEST=not-applicable", command)
        self.assertIn(
            "XSCIENTIST_RUNTIME_SPEC=xscientist[research,zhipu] @ "
            f"git+https://example.invalid/XScientist.git@{'f' * 40}",
            command,
        )
        self.assertNotIn(str(workspace), "\n".join(str(item) for item in command))
        self.assertFalse(Path(context).exists())

    def test_source_checkout_build_still_passes_source_context_and_arguments(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = self._workspace(td)
            source_context = Path(td) / "source-checkout"
            (source_context / "xscientist").mkdir(parents=True)
            (source_context / "ai_scientist").mkdir()
            (source_context / "compat").mkdir()
            (source_context / ".git").mkdir()
            (source_context / "pyproject.toml").write_text(
                "[project]\nname = 'xscientist'\n", encoding="utf-8"
            )
            (source_context / "README.md").write_text("source\n", encoding="utf-8")
            (source_context / "xscientist" / "runtime.py").write_text(
                "VALUE = 1\n", encoding="utf-8"
            )
            (source_context / "xscientist" / "private-token.txt").write_text(
                "must not reach Docker", encoding="utf-8"
            )
            (source_context / "source-sentinel.txt").write_text(
                "must not reach Docker", encoding="utf-8"
            )
            snapshot = executor_manager._source_context_snapshot(source_context)
            source_args = [
                "--build-arg",
                "XSCIENTIST_INSTALL_MODE=local",
                "--build-arg",
                f"XSCIENTIST_SOURCE_REVISION={'a' * 40}",
                "--build-arg",
                "XSCIENTIST_INSTALL_SOURCE=local-source",
            ]
            observed: dict[str, object] = {}

            def run(command, **kwargs):
                observed["command"] = list(command)
                observed["cwd"] = Path(kwargs["cwd"])
                context = Path(command[-1])
                observed["context"] = context
                observed["context_files"] = sorted(
                    path.relative_to(context).as_posix()
                    for path in context.rglob("*")
                    if path.is_file()
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            ready = {"ok": True, "image": f"xscientist-exec:{__version__}"}
            with (
                mock.patch(
                    "xscientist.executor_manager.shutil.which",
                    return_value="/usr/bin/docker",
                ),
                mock.patch(
                    "xscientist.executor_manager._expected_executor_identity",
                    return_value=executor_manager._ExecutorIdentity(
                        install_source="local-source",
                        revision="a" * 40,
                        source_root=source_context,
                        source_digest=snapshot.digest,
                    ),
                ),
                mock.patch(
                    "xscientist.executor_manager._source_checkout_state",
                    return_value=("a" * 40, snapshot.digest),
                ),
                mock.patch(
                    "xscientist.executor_manager.inspect_executor",
                    return_value=ready,
                ),
            ):
                payload = build_executor(workspace, run=run)

        command = observed["command"]
        context = observed["context"]
        self.assertEqual(payload, ready)
        self.assertNotEqual(observed["cwd"], source_context.resolve())
        self.assertEqual(observed["cwd"], context)
        self.assertEqual(Path(command[-1]), context)
        self.assertEqual(
            Path(command[command.index("-f") + 1]),
            context / "Dockerfile.executor",
        )
        self.assertIn("pyproject.toml", observed["context_files"])
        self.assertIn("xscientist/runtime.py", observed["context_files"])
        self.assertNotIn("source-sentinel.txt", observed["context_files"])
        self.assertNotIn("xscientist/private-token.txt", observed["context_files"])
        for argument in source_args:
            self.assertIn(argument, command)
        self.assertIn(
            "XSCIENTIST_RUNTIME_SPEC=/tmp/xscientist-build-context[research,zhipu]",
            command,
        )
        self.assertIn(f"XSCIENTIST_SOURCE_DIGEST={snapshot.digest}", command)
        self.assertNotIn(str(source_context), "\n".join(str(item) for item in command))
        self.assertFalse(Path(context).exists())

    def test_check_requires_exact_version_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = self._workspace(td)
            recipe_digest = self._recipe_digest(workspace)

            def run(command, **_kwargs):
                if command[1] == "info":
                    return subprocess.CompletedProcess(command, 0, '"27.0"\n', "")
                labels = {
                    "org.opencontainers.image.version": __version__,
                    "org.opencontainers.image.revision": "release",
                    "org.xscientist.install-source": "pypi-release",
                    "org.xscientist.executor-recipe": recipe_digest,
                }
                return subprocess.CompletedProcess(
                    command, 0, self._identity_payload(labels), ""
                )

            with (
                mock.patch(
                    "xscientist.executor_manager.shutil.which",
                    return_value="/usr/bin/docker",
                ),
                mock.patch(
                    "xscientist.executor_manager._expected_executor_identity",
                    return_value=executor_manager._ExecutorIdentity(
                        install_source="pypi-release",
                        revision="release",
                    ),
                ),
            ):
                status = inspect_executor(workspace, run=run)

        self.assertTrue(status["ok"])
        self.assertTrue(status["version_match"])
        self.assertTrue(status["install_source_match"])
        self.assertTrue(status["revision_match"])
        self.assertTrue(status["source_digest_match"])
        self.assertTrue(status["recipe_match"])
        self.assertEqual(status["image_id"], self._IMAGE_ID)
        self.assertEqual(status["install_source"], "pypi-release")
        self.assertEqual(status["expected_revision"], "release")
        self.assertFalse(status["host_paths_disclosed"])

    def test_recipe_label_tracks_dockerfile_capabilities_and_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = self._workspace(td)
            original_digest = self._recipe_digest(workspace)

            def inspect_with_recipe(recipe_digest):
                def run(command, **_kwargs):
                    if command[1] == "info":
                        return subprocess.CompletedProcess(command, 0, '"27.0"\n', "")
                    labels = {
                        "org.opencontainers.image.version": __version__,
                        "org.opencontainers.image.revision": "release",
                        "org.xscientist.install-source": "pypi-release",
                    }
                    if recipe_digest is not None:
                        labels["org.xscientist.executor-recipe"] = recipe_digest
                    return subprocess.CompletedProcess(
                        command, 0, self._identity_payload(labels), ""
                    )

                with (
                    mock.patch(
                        "xscientist.executor_manager.shutil.which",
                        return_value="/usr/bin/docker",
                    ),
                    mock.patch(
                        "xscientist.executor_manager._expected_executor_identity",
                        return_value=executor_manager._ExecutorIdentity(
                            install_source="pypi-release",
                            revision="release",
                        ),
                    ),
                ):
                    return inspect_executor(workspace, run=run)

            missing = inspect_with_recipe(None)
            wrong = inspect_with_recipe("0" * 64)
            dockerfile = workspace / "Dockerfile.executor"
            dockerfile.write_text(
                dockerfile.read_text(encoding="utf-8").replace(
                    "research,zhipu", "research,openai"
                ),
                encoding="utf-8",
            )
            changed_digest = self._recipe_digest(workspace)

        self.assertFalse(missing["ok"])
        self.assertIn("missing its build recipe identity", missing["error"])
        self.assertFalse(wrong["ok"])
        self.assertIn("does not match Dockerfile.executor", wrong["error"])
        self.assertNotEqual(original_digest, changed_digest)

    def test_pypi_build_arguments_override_legacy_runtime_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = self._workspace(td)
            recipe = executor_manager._executor_recipe(workspace)
            arguments = executor_manager._executor_build_arguments(
                executor_manager._ExecutorIdentity(
                    install_source="pypi-release",
                    revision="release",
                ),
                recipe,
            )
            materialized = executor_manager._materialized_dockerfile(recipe)

        self.assertIn(f"XSCIENTIST_VERSION={__version__}", arguments)
        self.assertIn("XSCIENTIST_SOURCE_REVISION=release", arguments)
        self.assertIn("XSCIENTIST_INSTALL_SOURCE=pypi-release", arguments)
        self.assertIn(
            f"XSCIENTIST_RUNTIME_SPEC=xscientist[research,zhipu]=={__version__}",
            arguments,
        )
        self.assertIn(f"XSCIENTIST_RECIPE_DIGEST={recipe.digest}", arguments)
        self.assertIn("XSCIENTIST_SOURCE_DIGEST=not-applicable", arguments)
        self.assertNotIn("xscientist[research,zhipu]==0.1.4", materialized)
        self.assertIn("$XSCIENTIST_RUNTIME_SPEC", materialized)

    def test_unreproducible_installed_origin_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            installed_module = (
                Path(td) / "site-packages" / "xscientist" / "executor_manager.py"
            )
            with (
                mock.patch(
                    "xscientist.executor_manager.__file__", str(installed_module)
                ),
                mock.patch(
                    "xscientist.onboarding._installed_runtime_source",
                    return_value=SimpleNamespace(
                        install_source="unreproducible",
                        revision="unknown",
                        source_url=None,
                        reproducible=False,
                        error="direct_url exists but is not reproducible",
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    executor_manager.ExecutorManagerError,
                    "direct_url exists but is not reproducible",
                ):
                    executor_manager._expected_executor_identity()

    def test_missing_or_wrong_revision_is_rejected_with_rebuild_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = self._workspace(td)
            recipe_digest = self._recipe_digest(workspace)

            def inspect_with_revision(revision, source="pypi-release"):
                def run(command, **_kwargs):
                    if command[1] == "info":
                        return subprocess.CompletedProcess(command, 0, '"27.0"\n', "")
                    labels = {
                        "org.opencontainers.image.version": __version__,
                        "org.xscientist.install-source": source,
                        "org.xscientist.executor-recipe": recipe_digest,
                    }
                    if revision is not None:
                        labels["org.opencontainers.image.revision"] = revision
                    return subprocess.CompletedProcess(
                        command, 0, self._identity_payload(labels), ""
                    )

                with (
                    mock.patch(
                        "xscientist.executor_manager.shutil.which",
                        return_value="/usr/bin/docker",
                    ),
                    mock.patch(
                        "xscientist.executor_manager._expected_executor_identity",
                        return_value=executor_manager._ExecutorIdentity(
                            install_source="pypi-release",
                            revision="release",
                        ),
                    ),
                ):
                    return inspect_executor(workspace, run=run)

            missing = inspect_with_revision(None)
            wrong = inspect_with_revision("old-release")
            wrong_source = inspect_with_revision("release", "vcs-commit")

        self.assertFalse(missing["ok"])
        self.assertFalse(missing["revision_match"])
        self.assertIn("missing its source revision", missing["error"])
        self.assertFalse(wrong["ok"])
        self.assertIn("old-release", wrong["error"])
        self.assertIn("expected release", wrong["error"])
        self.assertIn("executor prepare", wrong["next_action"])
        self.assertFalse(wrong_source["ok"])
        self.assertFalse(wrong_source["install_source_match"])
        self.assertIn("expected pypi-release", wrong_source["error"])

    def test_expected_identity_tracks_vcs_commit_and_checkout_dirty_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            installed_module = (
                Path(td) / "site-packages" / "xscientist" / "executor_manager.py"
            )
            with (
                mock.patch(
                    "xscientist.executor_manager.__file__", str(installed_module)
                ),
                mock.patch(
                    "xscientist.onboarding._installed_runtime_source",
                    return_value=SimpleNamespace(
                        install_source="vcs-commit",
                        revision="a" * 40,
                        source_url="https://example.invalid/XScientist.git",
                        reproducible=True,
                        error="",
                    ),
                ),
            ):
                installed = executor_manager._expected_executor_identity()
            with (
                mock.patch(
                    "xscientist.executor_manager.__file__", str(installed_module)
                ),
                mock.patch(
                    "xscientist.onboarding._installed_runtime_source",
                    return_value=SimpleNamespace(
                        install_source="pypi-release",
                        revision="release",
                        source_url=None,
                        reproducible=True,
                        error="",
                    ),
                ),
            ):
                release = executor_manager._expected_executor_identity()

            source_context = Path(td) / "source-checkout"
            (source_context / "xscientist").mkdir(parents=True)
            (source_context / "ai_scientist").mkdir()
            (source_context / "compat").mkdir()
            (source_context / ".git").mkdir()
            (source_context / "pyproject.toml").write_text(
                "[project]\nname = 'xscientist'\n", encoding="utf-8"
            )
            runtime = source_context / "xscientist" / "runtime.py"
            runtime.write_text("VALUE = 1\n", encoding="utf-8")

            def git_run(command, **_kwargs):
                if command[1:3] == ["rev-parse", "HEAD"]:
                    return subprocess.CompletedProcess(command, 0, "b" * 40 + "\n", "")
                return subprocess.CompletedProcess(
                    command, 0, " M xscientist/runtime.py\n", ""
                )

            with (
                mock.patch(
                    "xscientist.executor_manager.__file__",
                    str(source_context / "xscientist" / "executor_manager.py"),
                ),
                mock.patch(
                    "xscientist.executor_manager.subprocess.run",
                    side_effect=git_run,
                ),
            ):
                first_checkout = executor_manager._expected_executor_identity()
                runtime.write_text("VALUE = 2\n", encoding="utf-8")
                second_checkout = executor_manager._expected_executor_identity()

        self.assertEqual(installed.install_source, "vcs-commit")
        self.assertEqual(installed.revision, "a" * 40)
        self.assertEqual(installed.source_url, "https://example.invalid/XScientist.git")
        self.assertIsNone(installed.source_root)
        self.assertEqual(release.install_source, "pypi-release")
        self.assertEqual(release.revision, "release")
        self.assertEqual(first_checkout.install_source, "local-source")
        self.assertRegex(
            first_checkout.revision,
            rf"^{'b' * 40}-dirty\.[0-9a-f]{{16}}$",
        )
        self.assertNotEqual(first_checkout.revision, second_checkout.revision)
        self.assertNotEqual(first_checkout.source_digest, second_checkout.source_digest)
        self.assertEqual(first_checkout.source_root, source_context.resolve())

    def test_clean_checkout_digest_rejects_an_ignored_allowlisted_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = self._workspace(td)
            source_context = Path(td) / "source-checkout"
            (source_context / "xscientist").mkdir(parents=True)
            (source_context / "ai_scientist").mkdir()
            (source_context / "compat").mkdir()
            (source_context / "pyproject.toml").write_text(
                "[project]\nname = 'xscientist'\n", encoding="utf-8"
            )
            runtime = source_context / "xscientist" / "ignored_runtime.py"
            runtime.write_text("VALUE = 1\n", encoding="utf-8")

            def git_run(command, **_kwargs):
                if command[1:3] == ["rev-parse", "HEAD"]:
                    return subprocess.CompletedProcess(command, 0, "c" * 40 + "\n", "")
                # Simulate an allowlisted file hidden by a git-ignore rule.
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                mock.patch(
                    "xscientist.executor_manager.__file__",
                    str(source_context / "xscientist" / "executor_manager.py"),
                ),
                mock.patch(
                    "xscientist.executor_manager.subprocess.run",
                    side_effect=git_run,
                ),
            ):
                first = executor_manager._expected_executor_identity()
                runtime.write_text("VALUE = 2\n", encoding="utf-8")
                current = executor_manager._expected_executor_identity()

            recipe_digest = self._recipe_digest(workspace)

            def docker_run(command, **_kwargs):
                if command[1] == "info":
                    return subprocess.CompletedProcess(command, 0, '"27.0"\n', "")
                labels = {
                    "org.opencontainers.image.version": __version__,
                    "org.opencontainers.image.revision": current.revision,
                    "org.xscientist.install-source": "local-source",
                    "org.xscientist.source-digest": first.source_digest,
                    "org.xscientist.executor-recipe": recipe_digest,
                }
                return subprocess.CompletedProcess(
                    command, 0, self._identity_payload(labels), ""
                )

            with (
                mock.patch(
                    "xscientist.executor_manager.shutil.which",
                    return_value="/usr/bin/docker",
                ),
                mock.patch(
                    "xscientist.executor_manager._expected_executor_identity",
                    return_value=current,
                ),
            ):
                status = inspect_executor(workspace, run=docker_run)

        self.assertEqual(first.revision, "c" * 40)
        self.assertEqual(current.revision, first.revision)
        self.assertNotEqual(first.source_digest, current.source_digest)
        self.assertFalse(status["ok"])
        self.assertFalse(status["source_digest_match"])
        self.assertIn("controlled snapshot", status["error"])

    def test_identity_miss_rebuilds_instead_of_hitting_cache(self) -> None:
        stale = {
            "schema": "xscientist.executor-status.v1",
            "ok": False,
            "image": f"xscientist-exec:{__version__}",
            "error": "executor source revision old does not match expected new",
        }
        ready = {**stale, "ok": True, "error": None}
        with (
            mock.patch(
                "xscientist.executor_manager.inspect_executor", return_value=stale
            ),
            mock.patch(
                "xscientist.executor_manager.build_executor", return_value=ready
            ) as build,
        ):
            payload = prepare_executor(".")

        self.assertTrue(payload["ok"])
        self.assertFalse(payload["cache_hit"])
        self.assertTrue(payload["built"])
        build.assert_called_once_with(".", pull_base=False, run=subprocess.run)

    def test_prepare_reuses_a_valid_cached_image(self) -> None:
        ready = {
            "schema": "xscientist.executor-status.v1",
            "ok": True,
            "image": f"xscientist-exec:{__version__}",
        }
        with (
            mock.patch(
                "xscientist.executor_manager.inspect_executor", return_value=ready
            ),
            mock.patch("xscientist.executor_manager.build_executor") as build,
        ):
            payload = prepare_executor(".")

        self.assertTrue(payload["cache_hit"])
        self.assertFalse(payload["built"])
        build.assert_not_called()

    def test_missing_docker_gives_install_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = self._workspace(td)
            with mock.patch(
                "xscientist.executor_manager.shutil.which", return_value=None
            ):
                status = inspect_executor(workspace)

        self.assertFalse(status["ok"])
        self.assertFalse(status["docker_available"])
        self.assertIn(DOCKER_INSTALL_URL, status["error"])
        self.assertEqual(status["next_action"], DOCKER_INSTALL_URL)

    def test_initialized_workspace_with_missing_recipe_is_not_generic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            initialized = Path(td) / "initialized"
            generic = Path(td) / "generic"
            for root in (initialized, generic):
                root.mkdir()
                (root / "bfts_config.yaml").write_text("exec: {}\n")
            provider_state = initialized / ".xscientist" / "providers.json"
            provider_state.parent.mkdir()
            provider_state.write_text("{}\n")

            resolved_initialized = executor_manager.resolve_executor_workspace(
                initialized
            )
            resolved_generic = executor_manager.resolve_executor_workspace(generic)

        self.assertEqual(resolved_initialized, initialized.resolve())
        self.assertIsNone(resolved_generic)

    def test_executor_identity_probes_have_bounded_timeouts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = self._workspace(td)
            identity = executor_manager._ExecutorIdentity(
                install_source="pypi-release",
                revision="release",
            )
            for stage in ("daemon", "image"):
                calls: list[tuple[list[str], int | None]] = []

                def run(command, **kwargs):
                    calls.append((list(command), kwargs.get("timeout")))
                    if stage == "daemon" or command[1:3] == ["image", "inspect"]:
                        raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))
                    return subprocess.CompletedProcess(command, 0, '"27.0"\n', "")

                with (
                    self.subTest(stage=stage),
                    mock.patch(
                        "xscientist.executor_manager.shutil.which",
                        return_value="/usr/bin/docker",
                    ),
                    mock.patch(
                        "xscientist.executor_manager._expected_executor_identity",
                        return_value=identity,
                    ),
                ):
                    status = inspect_executor(workspace, run=run)

                self.assertFalse(status["ok"])
                self.assertIn("timed out", status["error"])
                self.assertEqual(calls[0][1], 5)
                if stage == "image":
                    self.assertEqual(calls[1][1], 10)

    def test_cli_check_explains_an_unavailable_executor(self) -> None:
        unavailable = {
            "schema": "xscientist.executor-status.v1",
            "ok": False,
            "image": f"xscientist-exec:{__version__}",
            "daemon_ready": False,
            "image_available": False,
            "version_match": False,
            "install_source": None,
            "next_action": "xscientist executor prepare --workspace study",
        }
        output = io.StringIO()
        with (
            mock.patch(
                "xscientist.executor_manager.inspect_executor", return_value=unavailable
            ),
            contextlib.redirect_stdout(output),
        ):
            exit_code = cli_main(["executor", "check", "--workspace", "study"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Image available: False", output.getvalue())
        self.assertIn("executor prepare", output.getvalue())


if __name__ == "__main__":
    unittest.main()
