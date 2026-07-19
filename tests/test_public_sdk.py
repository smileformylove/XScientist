from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
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
from xscientist.entrypoints import project_main
from xscientist.service import run_server


class PublicSdkTests(unittest.TestCase):
    def test_top_level_api_is_stable_and_lightweight(self) -> None:
        self.assertEqual(xscientist.__version__, "0.2.0")
        self.assertIs(xscientist.XScientist, XScientist)
        self.assertTrue(callable(xscientist.create_app))

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

    def test_project_request_requires_an_input_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "topic or ideas"):
            ProjectRequest(project="demo").to_argv()

    def test_client_uses_installed_module_entrypoint_and_packaged_config(self) -> None:
        client = XScientist(output_root="/tmp/xscientist-output")
        command = client.project_command(
            ProjectRequest(project="demo", topic="topic.md")
        )

        self.assertEqual(command[1:3], ["-m", "run_project"])
        self.assertIn("--output-root", command)
        config_path = Path(command[command.index("--bfts-config") + 1])
        self.assertTrue(config_path.is_file())

    def test_client_captures_process_result(self) -> None:
        client = XScientist()
        result = client.run_command([client.python_executable, "-c", "print('ready')"])

        self.assertTrue(result.ok)
        self.assertEqual(result.stdout.strip(), "ready")

    def test_cli_info_has_machine_readable_output(self) -> None:
        with mock.patch("builtins.print") as printer:
            exit_code = cli_main(["info", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(printer.call_args.args[0])
        self.assertEqual(payload["name"], "xscientist")

    def test_workflow_entrypoint_reports_missing_full_dependencies_cleanly(
        self,
    ) -> None:
        real_import = __import__("importlib").import_module

        def fail_run_project(name: str, package=None):
            if name == "run_project":
                error = ModuleNotFoundError("No module named 'numpy'")
                error.name = "numpy"
                raise error
            return real_import(name, package)

        with (
            mock.patch(
                "xscientist.entrypoints.importlib.import_module",
                side_effect=fail_run_project,
            ),
            mock.patch("sys.stderr") as stderr,
        ):
            exit_code = project_main(["--help"])

        self.assertEqual(exit_code, 2)
        output = "".join(
            str(call.args[0]) for call in stderr.write.call_args_list if call.args
        )
        self.assertIn("xscientist[full]", output)

    def test_service_settings_validate_workers(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_workers"):
            ServiceSettings(max_workers=0)

    def test_service_truncates_large_process_output(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")

        completed = xscientist.CommandResult(
            command=("python", "-m", "run_project"),
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
            self.assertEqual(client.get("/health").status_code, 401)
            response = client.get("/health", headers={"X-API-Key": "secret"})

        self.assertEqual(response.status_code, 200)

    def test_reload_server_does_not_create_a_second_app(self) -> None:
        with (
            mock.patch("uvicorn.run") as uvicorn_run,
            mock.patch("xscientist.service.create_app") as create_app_mock,
        ):
            run_server(reload=True)

        uvicorn_run.assert_called_once()
        create_app_mock.assert_not_called()

    def test_http_service_submits_and_reports_jobs(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("service extras not installed")

        completed = xscientist.CommandResult(
            command=("python", "-m", "run_project"),
            returncode=0,
            stdout="done",
            stderr="",
            started_at="start",
            finished_at="finish",
        )
        with mock.patch.object(XScientist, "run_project", return_value=completed):
            app = xscientist.create_app(ServiceSettings(max_workers=1))
            with TestClient(app) as client:
                response = client.post(
                    "/v1/projects",
                    json={"project": "demo", "topic": "topic.md"},
                )
                self.assertEqual(response.status_code, 202)
                job_id = response.json()["id"]
                for _ in range(50):
                    status = client.get(f"/v1/jobs/{job_id}").json()
                    if status["status"] != "queued" and status["status"] != "running":
                        break
                self.assertEqual(status["status"], "succeeded")
                self.assertEqual(status["result"]["stdout"], "done")


if __name__ == "__main__":
    unittest.main()
