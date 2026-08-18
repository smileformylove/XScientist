from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import time
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import xscientist
from ai_scientist.resources import (
    bfts_config_path,
    idea_resource_path,
    latex_template_dir,
    resolve_bfts_config_path,
)
from xscientist import ProjectRequest, ServiceSettings, XScientist
from xscientist.cli import main as cli_main
from xscientist.entrypoints import batch_main, project_main
from xscientist.service import run_server


class PublicSdkTests(unittest.TestCase):
    def test_top_level_api_is_stable_and_lightweight(self) -> None:
        self.assertEqual(xscientist.__version__, "0.1.3")
        self.assertIs(xscientist.XScientist, XScientist)
        self.assertTrue(callable(xscientist.create_app))
        self.assertTrue(callable(xscientist.build_research_dag))
        self.assertTrue(callable(xscientist.build_research_guide))

    def test_packaged_runtime_resources_exist(self) -> None:
        self.assertTrue(bfts_config_path("default").is_file())
        self.assertTrue(bfts_config_path("deep").is_file())
        self.assertTrue(idea_resource_path().is_file())
        self.assertTrue(
            idea_resource_path("i_cant_believe_its_not_better.md").is_file()
        )
        self.assertTrue((latex_template_dir("icbinb") / "template.tex").is_file())
        self.assertTrue((latex_template_dir("icml") / "template.tex").is_file())

    def test_config_alias_falls_back_to_packaged_profile(self) -> None:
        missing_root_config = Path("/definitely/missing/bfts_config.yaml")
        self.assertEqual(
            resolve_bfts_config_path(missing_root_config),
            bfts_config_path("default"),
        )

    def test_source_checkout_alias_uses_repository_config(self) -> None:
        resolved = resolve_bfts_config_path("bfts_config.yaml")

        self.assertEqual(resolved.name, "bfts_config.yaml")
        self.assertEqual(resolved.parent.name, "bfts")
        self.assertEqual(resolved.parent.parent.name, "configs")

    def test_relative_custom_config_resolves_from_current_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "custom.yaml"
            config.write_text("goal: demo\n", encoding="utf-8")
            with mock.patch("pathlib.Path.cwd", return_value=root):
                resolved = resolve_bfts_config_path("custom.yaml")

            self.assertEqual(resolved, config.resolve())

    def test_client_resolves_custom_config_from_work_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "custom.yaml"
            config.write_text("goal: demo\n", encoding="utf-8")
            client = XScientist(work_dir=root)

            command = client.project_command(
                ProjectRequest(
                    project="demo",
                    topic="topic.md",
                    bfts_config="custom.yaml",
                )
            )

            self.assertEqual(
                Path(command[command.index("--bfts-config") + 1]),
                config.resolve(),
            )

    def test_project_request_builds_predictable_cli_arguments(self) -> None:
        request = ProjectRequest(
            project="demo",
            topic="topic.md",
            num_ideas=2,
            parallel=True,
            num_workers=3,
            workflow_mode="program_driven",
            high_quality_mode=True,
        )

        argv = request.to_argv()

        self.assertEqual(argv[0], "demo")
        self.assertIn("--topic", argv)
        self.assertIn("--parallel", argv)
        self.assertIn("--high-quality-mode", argv)
        self.assertEqual(argv[argv.index("--num-workers") + 1], "3")

    def test_project_request_exposes_guarded_autopilot_contract(self) -> None:
        request = ProjectRequest(
            project="demo",
            question="Does X affect Y?",
            autopilot="discovery",
            resume=True,
            allow_synthetic_data=True,
            max_project_tokens=100_000,
            max_cost_usd=10,
        )

        argv = request.to_argv()

        self.assertEqual(argv[argv.index("--question") + 1], "Does X affect Y?")
        self.assertEqual(argv[argv.index("--autopilot") + 1], "discovery")
        self.assertIn("--resume", argv)
        self.assertIn("--allow-synthetic-data", argv)
        self.assertEqual(argv[argv.index("--max-cost-usd") + 1], "10")

    def test_project_request_can_enable_local_research_git_without_remote(self) -> None:
        request = ProjectRequest(
            project="demo",
            topic="topic.md",
            research_git="local",
            git_checkpoint_policy="stage",
            research_git_strict=True,
        )

        argv = request.to_argv()

        self.assertEqual(argv[argv.index("--research-git") + 1], "local")
        self.assertEqual(argv[argv.index("--git-checkpoint-policy") + 1], "stage")
        self.assertIn("--research-git-strict", argv)

    def test_project_request_exposes_native_research_vcs_spelling(self) -> None:
        request = ProjectRequest(
            project="demo",
            topic="topic.md",
            research_git="off",
            research_vcs="local",
            git_checkpoint_policy="stage",
            research_git_strict=True,
        )

        argv = request.to_argv()

        self.assertEqual(argv[argv.index("--research-vcs") + 1], "local")
        self.assertEqual(argv[argv.index("--checkpoint-policy") + 1], "stage")
        self.assertIn("--research-vcs-strict", argv)

    def test_project_request_requires_an_input_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "topic or ideas"):
            ProjectRequest(project="demo").to_argv()

    def test_client_uses_installed_module_entrypoint_and_packaged_config(self) -> None:
        client = XScientist(output_root="/tmp/xscientist-output")
        command = client.project_command(
            ProjectRequest(project="demo", topic="topic.md")
        )

        self.assertEqual(command[1:3], ["-m", "ai_scientist.apps.project"])
        self.assertIn("--output-root", command)
        config_path = Path(command[command.index("--bfts-config") + 1])
        self.assertTrue(config_path.is_file())

    def test_client_captures_process_result(self) -> None:
        client = XScientist()
        result = client.run_command([client.python_executable, "-c", "print('ready')"])

        self.assertTrue(result.ok)
        self.assertEqual(result.stdout.strip(), "ready")

    def test_client_exposes_read_only_research_views(self) -> None:
        manager = mock.Mock()
        manager.papers_dir = Path("/tmp/xscientist-output/papers")
        manager.list_papers.return_value = [{"name": "paper-a"}, {"name": "paper-b"}]
        manager.get_paper_details.return_value = {"folder": "paper-a"}
        manager.shortlist_papers.return_value = [{"name": "paper-a"}]
        manager.submission_board.return_value = {"iclr": [{"name": "paper-a"}]}
        manager.rewrite_board.return_value = [{"name": "paper-b"}]
        client = XScientist(output_root="/tmp/xscientist-output")

        with mock.patch.object(client, "_research_manager", return_value=manager):
            self.assertEqual(client.list_papers(limit=1), [{"name": "paper-a"}])
            self.assertEqual(client.get_paper("paper-a"), {"folder": "paper-a"})
            self.assertEqual(client.shortlist_papers(top_n=2), [{"name": "paper-a"}])
            self.assertEqual(
                client.submission_board(top_n_per_venue=2),
                {"iclr": [{"name": "paper-a"}]},
            )
            self.assertEqual(client.rewrite_board(top_n=2), [{"name": "paper-b"}])

        manager.list_papers.assert_called_once_with(
            paper_type=None,
            sort_by="modified",
        )
        manager.get_paper_details.assert_called_once_with("paper-a")
        manager.shortlist_papers.assert_called_once()
        manager.submission_board.assert_called_once()
        manager.rewrite_board.assert_called_once()

    def test_client_read_only_views_validate_limits(self) -> None:
        client = XScientist()
        with mock.patch.object(client, "_research_manager") as manager:
            for kwargs in ({"limit": 0}, {"limit": 1001}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    client.list_papers(**kwargs)
            with self.assertRaisesRegex(ValueError, "sort_by"):
                client.list_papers(sort_by="unknown")
            with self.assertRaises(ValueError):
                client.shortlist_papers(top_n=0)
            with self.assertRaises(ValueError):
                client.submission_board(top_n_per_venue=0)
            with self.assertRaises(ValueError):
                client.rewrite_board(top_n=0)

        manager.assert_not_called()

    def test_client_paper_details_reject_path_escape_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "output"
            papers_root = output_root / "papers"
            papers_root.mkdir(parents=True)
            outside = Path(td) / "outside"
            outside.mkdir()
            (papers_root / "linked").symlink_to(outside, target_is_directory=True)
            client = XScientist(output_root=output_root)

            for folder in ("../escape", "nested/paper", "~", " linked ", "linked"):
                with self.subTest(folder=folder), self.assertRaises(ValueError):
                    client.get_paper(folder)

    def test_cli_info_has_machine_readable_output(self) -> None:
        with mock.patch("builtins.print") as printer:
            exit_code = cli_main(["info", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(printer.call_args.args[0])
        self.assertEqual(payload["name"], "xscientist")
        self.assertIn(
            payload["installation_profile"],
            {"core", "core+service", "research", "research+service"},
        )
        self.assertIsInstance(payload["research_runtime_ready"], bool)
        self.assertIsInstance(payload["service_ready"], bool)
        self.assertIsInstance(payload["missing_research_packages"], list)
        self.assertIn(payload["provider_client_ready"], {True, False, None})
        self.assertIn(
            payload["provider_client_status"],
            {"ready", "missing", "not_configured"},
        )
        self.assertTrue(
            payload["missing_provider_clients"] is None
            or isinstance(payload["missing_provider_clients"], list)
        )
        self.assertIn("suggested_provider", payload)
        self.assertIn("xscientist[research", payload["recommended_install"])
        self.assertIn(payload["output_root"], {"<configured>", "<default>"})
        self.assertFalse(Path(payload["python_executable"]).is_absolute())
        self.assertFalse(payload["host_paths_disclosed"])
        self.assertEqual(
            payload["quickstart"],
            "xscientist demo ./xscientist-demo --autopilot --open",
        )

    def test_cli_info_human_output_does_not_expose_internal_null_fields(self) -> None:
        payload = {
            "version": "0.1.3",
            "installation_profile": "core",
            "research_runtime_ready": False,
            "missing_research_packages": ["sklearn"],
            "provider_configured": False,
            "active_provider": None,
            "provider_client_status": "not_configured",
            "suggested_provider": "ollama",
            "discovered_local_models": ["ollama/qwen2.5:7b"],
            "recommended_install": 'python -m pip install "xscientist[research,openai-compatible]"',
            "quickstart": "xscientist demo ./xscientist-demo --autopilot --open",
        }
        output = io.StringIO()
        with (
            mock.patch("xscientist.cli._installation_info", return_value=payload),
            contextlib.redirect_stdout(output),
        ):
            exit_code = cli_main(["info"])

        self.assertEqual(exit_code, 0)
        self.assertIn("detected 1 Ollama model", output.getvalue())
        self.assertIn("Suggested model: ollama/qwen2.5:7b", output.getvalue())
        self.assertNotIn("provider_client_ready", output.getvalue())

    def test_cli_forwards_workflow_arguments_without_parsing_them(self) -> None:
        project = mock.Mock(return_value=0)
        with mock.patch.dict("xscientist.cli._DELEGATES", {"project": project}):
            exit_code = cli_main(
                ["project", "demo", "--topic", "topic.md", "--num-ideas", "2"]
            )

        self.assertEqual(exit_code, 0)
        project.assert_called_once_with(
            ["demo", "--topic", "topic.md", "--num-ideas", "2"]
        )

    def test_cli_forwards_workflow_help_to_the_selected_entrypoint(self) -> None:
        project = mock.Mock(return_value=0)
        with mock.patch.dict("xscientist.cli._DELEGATES", {"project": project}):
            exit_code = cli_main(["project", "--help"])

        self.assertEqual(exit_code, 0)
        project.assert_called_once_with(["--help"])

    def test_workflow_help_does_not_require_full_dependencies(
        self,
    ) -> None:
        real_import = __import__("importlib").import_module

        def fail_full_workflows(name: str, package=None):
            if name in {"ai_scientist.apps.project", "ai_scientist.apps.batch"}:
                error = ModuleNotFoundError("No module named 'numpy'")
                error.name = "numpy"
                raise error
            return real_import(name, package)

        stdout = io.StringIO()
        with (
            mock.patch.object(sys, "stdout", stdout),
            mock.patch(
                "xscientist.entrypoints.importlib.import_module",
                side_effect=fail_full_workflows,
            ),
        ):
            project_exit_code = project_main(["--help"])
            batch_exit_code = batch_main(["--help"])

        self.assertEqual(project_exit_code, 0)
        self.assertEqual(batch_exit_code, 0)
        self.assertIn("usage: xscientist project", stdout.getvalue())
        self.assertIn("usage: xscientist batch", stdout.getvalue())

    def test_workflow_entrypoint_reports_missing_full_dependencies_cleanly(
        self,
    ) -> None:
        error = ModuleNotFoundError("No module named 'numpy'")
        error.name = "numpy"
        with (
            mock.patch.object(sys, "stderr") as stderr,
            mock.patch(
                "xscientist.entrypoints.importlib.import_module",
                side_effect=error,
            ),
        ):
            exit_code = project_main(["demo", "--topic", "topic.md"])

        self.assertEqual(exit_code, 2)
        output = "".join(
            str(call.args[0]) for call in stderr.write.call_args_list if call.args
        )
        self.assertIn("xscientist[research]", output)

    def test_workflow_entrypoint_reports_expected_runtime_failures_without_traceback(
        self,
    ) -> None:
        module = SimpleNamespace(
            main=mock.Mock(side_effect=RuntimeError("isolated executor unavailable"))
        )
        stderr = io.StringIO()
        with (
            mock.patch(
                "xscientist.entrypoints.importlib.import_module", return_value=module
            ),
            mock.patch.object(sys, "stderr", stderr),
        ):
            exit_code = project_main(["demo", "--topic", "topic.md"])

        self.assertEqual(exit_code, 2)
        self.assertIn("project stopped", stderr.getvalue())
        self.assertIn("isolated executor unavailable", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_service_settings_validate_workers(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_workers"):
            ServiceSettings(max_workers=0)

    def test_service_validates_work_dir_at_startup_and_exposes_discovery_root(
        self,
    ) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")

        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing"
            with self.assertRaisesRegex(ValueError, "work_dir does not exist"):
                xscientist.create_app(ServiceSettings(work_dir=missing))

            app = xscientist.create_app(ServiceSettings(work_dir=td))
            with TestClient(app) as client:
                response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["documentation"], "/docs")

    def test_service_truncates_large_process_output(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")

        completed = xscientist.CommandResult(
            command=("python", "-m", "ai_scientist.apps.project"),
            returncode=0,
            stdout="0123456789",
            stderr="abcdefghij",
            started_at="start",
            finished_at="finish",
        )
        settings = ServiceSettings(max_workers=1, max_output_chars=4)
        with mock.patch.object(XScientist, "run_project", return_value=completed):
            app = xscientist.create_app(settings)
            with TestClient(app) as client:
                job_id = client.post(
                    "/v1/projects",
                    json={"project": "demo", "topic": "topic.md"},
                ).json()["id"]
                for _ in range(50):
                    payload = client.get(f"/v1/jobs/{job_id}").json()
                    if payload["status"] not in {"queued", "running"}:
                        break

        self.assertEqual(payload["result"]["stdout"], "6789")
        self.assertEqual(payload["result"]["stderr"], "ghij")

    def test_http_service_supports_optional_api_key(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")

        app = xscientist.create_app(ServiceSettings(max_workers=1, api_key="secret"))
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/v1/jobs").status_code, 401)
            response = client.get("/v1/jobs", headers={"X-API-Key": "secret"})

        self.assertEqual(response.status_code, 200)

    def test_non_loopback_service_requires_auth_or_explicit_override(self) -> None:
        uvicorn = mock.Mock()
        with mock.patch.dict("sys.modules", {"uvicorn": uvicorn}):
            with self.assertRaisesRegex(ValueError, "Non-loopback"):
                run_server(host="0.0.0.0")
        uvicorn.run.assert_not_called()

        app = mock.Mock()
        with (
            mock.patch.dict("sys.modules", {"uvicorn": uvicorn}),
            mock.patch("xscientist.service.create_app", return_value=app),
        ):
            run_server(host="0.0.0.0", allow_unauthenticated=True)
        uvicorn.run.assert_called_once_with(app, host="0.0.0.0", port=8000)

    def test_client_bounded_capture_retains_only_tail(self) -> None:
        client = XScientist()
        result = client.run_command(
            [sys.executable, "-c", "print('x' * 10000)"],
            max_output_chars=64,
        )

        self.assertTrue(result.ok)
        self.assertEqual(len(result.stdout), 64)
        self.assertTrue(result.stdout_truncated)

    def test_client_workspace_quota_terminates_process(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = XScientist(work_dir=root).run_command(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('large.bin').write_bytes(b'x' * 4096)",
                ],
                max_output_chars=256,
                workspace=root,
                max_workspace_bytes=64,
                max_workspace_files=10,
            )

        self.assertEqual(result.returncode, 75)
        self.assertIn("ResourceLimitError", result.stderr)

    @unittest.skipUnless(shutil.which("git"), "Git is required for Research VCS API")
    def test_http_service_exposes_payload_free_research_audit(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")
        from xscientist import ResearchRepository

        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "output"
            project = output_root / "projects" / "demo"
            repository = ResearchRepository.init(
                project,
                question="Q?",
                git_user_name="Research Test",
                git_user_email="research@example.invalid",
            )
            repository.record("claim", {"statement": "unbound claim"})
            repository.commit(stage="evidence", subject="record unbound claim")
            app = xscientist.create_app(
                ServiceSettings(max_workers=1, output_root=output_root)
            )

            with TestClient(app) as client:
                status = client.get("/v1/projects/demo/research/status")
                audit = client.get(
                    "/v1/projects/demo/research/audit", params={"level": "trace"}
                )
                dag = client.get("/v1/projects/demo/research/dag")
                escaped = client.get("/v1/projects/../research/audit")

            self.assertEqual(status.status_code, 200)
            self.assertEqual(audit.status_code, 200)
            self.assertFalse(audit.json()["complete"])
            self.assertFalse(audit.json()["payloads_disclosed"])
            self.assertEqual(dag.status_code, 200)
            self.assertFalse(dag.json()["content_disclosed"])
            self.assertNotIn("unbound claim", dag.text)
            self.assertTrue(dag.json()["integrity"]["is_dag"])
            self.assertNotIn(str(Path(td).resolve()), status.text)
            self.assertNotIn(str(Path(td).resolve()), dag.text)
            self.assertIn(escaped.status_code, {404, 422})

    def test_reload_server_does_not_create_a_second_app(self) -> None:
        uvicorn = mock.Mock()
        with (
            mock.patch.dict("sys.modules", {"uvicorn": uvicorn}),
            mock.patch("xscientist.service.create_app") as create_app_mock,
        ):
            run_server(reload=True)

        uvicorn.run.assert_called_once_with(
            "xscientist.service:create_app",
            factory=True,
            host="127.0.0.1",
            port=8000,
            reload=True,
        )
        create_app_mock.assert_not_called()

    def test_http_service_submits_and_reports_jobs(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")

        completed = xscientist.CommandResult(
            command=("python", "-m", "ai_scientist.apps.project"),
            returncode=0,
            stdout="done",
            stderr="",
            started_at="start",
            finished_at="finish",
        )
        with mock.patch.object(
            XScientist, "run_project", return_value=completed
        ) as run_project:
            app = xscientist.create_app(ServiceSettings(max_workers=1))
            with TestClient(app) as client:
                response = client.post(
                    "/v1/projects",
                    json={
                        "project": "demo",
                        "question": "Does X affect Y?",
                        "autopilot": "balanced",
                        "allow_synthetic_data": True,
                        "max_project_tokens": 100_000,
                    },
                )
                self.assertEqual(response.status_code, 202)
                job_id = response.json()["id"]
                for _ in range(50):
                    status = client.get(f"/v1/jobs/{job_id}").json()
                    if status["status"] != "queued" and status["status"] != "running":
                        break
                self.assertEqual(status["status"], "succeeded")
                self.assertEqual(status["result"]["stdout"], "done")

        submitted = run_project.call_args.args[0]
        self.assertEqual(submitted.question, "Does X affect Y?")
        self.assertEqual(submitted.autopilot, "balanced")
        self.assertTrue(submitted.allow_synthetic_data)

    def test_http_service_exposes_confined_read_only_research_views(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")

        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "output"
            output_root.mkdir()
            inside = output_root / "papers" / "demo"
            outside = Path(td) / "private.txt"
            views = {
                "papers": [
                    {"name": "demo", "path": str(inside), "private": str(outside)}
                ],
                "shortlist": [{"name": "demo", "path": str(inside)}],
                "submission": {"iclr": [{"name": "demo", "path": str(inside)}]},
                "rewrite": [{"name": "demo", "path": str(inside)}],
            }
            with (
                mock.patch.object(
                    XScientist, "list_papers", return_value=views["papers"]
                ),
                mock.patch.object(
                    XScientist,
                    "get_paper",
                    return_value={"folder": "demo", "path": str(inside)},
                ),
                mock.patch.object(
                    XScientist, "shortlist_papers", return_value=views["shortlist"]
                ),
                mock.patch.object(
                    XScientist, "submission_board", return_value=views["submission"]
                ),
                mock.patch.object(
                    XScientist, "rewrite_board", return_value=views["rewrite"]
                ),
            ):
                app = xscientist.create_app(
                    ServiceSettings(output_root=output_root, max_workers=1)
                )
                with TestClient(app) as client:
                    papers = client.get("/v1/papers?limit=10").json()
                    detail = client.get("/v1/papers/demo").json()
                    shortlist = client.get("/v1/shortlist?top_n=2").json()
                    submission = client.get(
                        "/v1/boards/submission?top_n_per_venue=2"
                    ).json()
                    rewrite = client.get("/v1/boards/rewrite?top_n=2").json()

            self.assertEqual(papers["items"][0]["path"], "papers/demo")
            self.assertIsNone(papers["items"][0]["private"])
            self.assertEqual(detail["paper"]["path"], "papers/demo")
            self.assertEqual(shortlist["count"], 1)
            self.assertEqual(submission["count"], 1)
            self.assertEqual(rewrite["count"], 1)

    def test_http_read_only_views_reject_invalid_limits(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")

        app = xscientist.create_app(ServiceSettings(max_workers=1))
        with TestClient(app) as client:
            responses = [
                client.get("/v1/papers?limit=0"),
                client.get("/v1/shortlist?top_n=1001"),
                client.get("/v1/boards/submission?top_n_per_venue=0"),
                client.get("/v1/boards/rewrite?top_n=0"),
            ]

        self.assertTrue(
            all(response.status_code == 422 for response in responses),
            [(response.status_code, response.text) for response in responses],
        )

    def test_http_paper_details_maps_invalid_and_missing_folders(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")

        app = xscientist.create_app(ServiceSettings(max_workers=1))
        with mock.patch.object(
            XScientist,
            "get_paper",
            side_effect=[ValueError("invalid folder"), None],
        ):
            with TestClient(app) as client:
                invalid = client.get("/v1/papers/invalid")
                missing = client.get("/v1/papers/missing")

        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(missing.status_code, 404)

    def test_http_service_confines_requests_to_service_directories(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")

        completed = xscientist.CommandResult(
            command=("python", "-m", "ai_scientist.apps.project"),
            returncode=0,
            stdout="done",
            stderr="",
            started_at="start",
            finished_at="finish",
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work_dir = root / "work"
            output_root = root / "output"
            work_dir.mkdir()
            (work_dir / "topic.md").write_text("topic", encoding="utf-8")
            app = xscientist.create_app(
                ServiceSettings(
                    work_dir=work_dir,
                    output_root=output_root,
                    max_workers=1,
                )
            )
            with (
                mock.patch.object(
                    XScientist, "run_project", return_value=completed
                ) as run_project,
                TestClient(app) as client,
            ):
                response = client.post(
                    "/v1/projects",
                    json={"project": "demo", "topic": "topic.md"},
                )
                self.assertEqual(response.status_code, 202, response.text)
                for _ in range(50):
                    job = client.get(f"/v1/jobs/{response.json()['id']}").json()
                    if job["status"] not in {"queued", "running"}:
                        break
                    time.sleep(0.01)

            request = run_project.call_args.args[0]
            self.assertEqual(request.project, "demo")
            self.assertEqual(Path(request.topic), (work_dir / "topic.md").resolve())
            self.assertEqual(Path(request.output_root), output_root.resolve())

    def test_http_service_rejects_path_and_argument_overrides(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work_dir = root / "work"
            output_root = root / "output"
            work_dir.mkdir()
            projects_root = output_root / "projects"
            projects_root.mkdir(parents=True)
            outside_project = root / "outside-project"
            outside_project.mkdir()
            (projects_root / "linked").symlink_to(
                outside_project, target_is_directory=True
            )
            app = xscientist.create_app(
                ServiceSettings(work_dir=work_dir, output_root=output_root)
            )
            payloads = [
                {"project": "../escape", "topic": "topic.md"},
                {"project": "nested/demo", "topic": "topic.md"},
                {"project": "~", "topic": "topic.md"},
                {"project": "--output-root", "topic": "topic.md"},
                {"project": " demo ", "topic": "topic.md"},
                {"project": "demo\n", "topic": "topic.md"},
                {"project": "linked", "topic": "topic.md"},
                {"project": "demo", "topic": str(root / "outside.md")},
                {
                    "project": "demo",
                    "topic": "topic.md",
                    "output_root": str(root / "other-output"),
                },
                {
                    "project": "demo",
                    "topic": "topic.md",
                    "extra_args": ["--output-root", str(root / "other-output")],
                },
                {
                    "project": "demo",
                    "topic": "topic.md",
                    "extra_args": ["--topic=../outside.md"],
                },
                {
                    "project": "demo",
                    "topic": "topic.md",
                    "extra_args": ["--seed-from-ara", str(root / "outside-ara")],
                },
                {
                    "project": "demo",
                    "topic": "topic.md",
                    "bfts_config": str(root / "outside.yaml"),
                },
            ]
            with TestClient(app) as client:
                responses = [
                    client.post("/v1/projects", json=payload) for payload in payloads
                ]

        self.assertTrue(
            all(response.status_code == 422 for response in responses),
            [(response.status_code, response.text) for response in responses],
        )

    def test_http_jobs_survive_service_restart(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")

        completed = xscientist.CommandResult(
            command=("python", "-m", "run_project"),
            returncode=0,
            stdout="persisted",
            stderr="",
            started_at="start",
            finished_at="finish",
        )
        with tempfile.TemporaryDirectory() as td:
            settings = ServiceSettings(max_workers=1, state_dir=td)
            with mock.patch.object(XScientist, "run_project", return_value=completed):
                first_app = xscientist.create_app(settings)
                with TestClient(first_app) as client:
                    job_id = client.post(
                        "/v1/projects",
                        json={"project": "demo", "topic": "topic.md"},
                    ).json()["id"]
                    for _ in range(50):
                        payload = client.get(f"/v1/jobs/{job_id}").json()
                        if payload["status"] == "succeeded":
                            break
                        time.sleep(0.01)

            self.assertTrue((Path(td) / f"{job_id}.json").is_file())
            second_app = xscientist.create_app(settings)
            with TestClient(second_app) as client:
                restored = client.get(f"/v1/jobs/{job_id}").json()

            self.assertEqual(restored["status"], "succeeded")
            self.assertEqual(restored["result"]["stdout"], "persisted")

    def test_service_marks_incomplete_restored_job_interrupted(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")

        with tempfile.TemporaryDirectory() as td:
            job_id = "unfinished"
            payload = {
                "id": job_id,
                "status": "running",
                "created_at": "created",
                "started_at": "started",
                "finished_at": None,
                "error": None,
                "request": ProjectRequest(project="demo", topic="topic.md").to_dict(),
                "result": None,
            }
            (Path(td) / f"{job_id}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            app = xscientist.create_app(ServiceSettings(max_workers=1, state_dir=td))
            with TestClient(app) as client:
                restored = client.get(f"/v1/jobs/{job_id}").json()

            self.assertEqual(restored["status"], "interrupted")
            self.assertIn("restarted", restored["error"])

    def test_service_rejects_job_when_initial_state_cannot_persist(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")

        with tempfile.TemporaryDirectory() as td:
            app = xscientist.create_app(ServiceSettings(max_workers=1, state_dir=td))
            with (
                mock.patch(
                    "xscientist.service.atomic_write_json",
                    side_effect=OSError("disk busy"),
                ),
                TestClient(app, raise_server_exceptions=False) as client,
            ):
                response = client.post(
                    "/v1/projects",
                    json={"project": "demo", "topic": "topic.md"},
                )
                jobs = client.get("/v1/jobs").json()["items"]

            self.assertEqual(response.status_code, 500)
            self.assertEqual(jobs, [])

    def test_worker_reports_running_transition_persistence_failure(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")

        with tempfile.TemporaryDirectory() as td:
            app = xscientist.create_app(ServiceSettings(max_workers=1, state_dir=td))
            from ai_scientist.utils.atomic_io import (
                atomic_write_json as real_atomic_write_json,
            )

            calls = 0

            def fail_second_write(path, payload):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("disk busy")
                return real_atomic_write_json(path, payload)

            with (
                mock.patch(
                    "xscientist.service.atomic_write_json",
                    side_effect=fail_second_write,
                ),
                TestClient(app) as client,
            ):
                job_id = client.post(
                    "/v1/projects",
                    json={"project": "demo", "topic": "topic.md"},
                ).json()["id"]
                for _ in range(50):
                    payload = client.get(f"/v1/jobs/{job_id}").json()
                    if payload["status"] == "failed":
                        break
                    time.sleep(0.01)

            self.assertEqual(payload["status"], "failed")
            self.assertIn("disk busy", payload["error"])


if __name__ == "__main__":
    unittest.main()
