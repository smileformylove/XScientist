from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import ai_scientist.resources as resources
from ai_scientist.resources import (
    OfficialTemplateError,
    materialize_latex_template,
    verify_latex_template_source,
)
from ai_scientist.utils.writeup_workflow import (
    build_writeup_execution_plan,
    resolve_page_limit,
)


class _MemoryResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, content_length: int | None = None) -> None:
        super().__init__(payload)
        self.status = 200
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)


def _zip_payload(
    files: dict[str, bytes],
    *,
    symlinks: set[str] | None = None,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            info = zipfile.ZipInfo(name)
            if name in (symlinks or set()):
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, content)
    return output.getvalue()


def _fixture_files(venue: str) -> dict[str, bytes]:
    spec = resources._OFFICIAL_TEMPLATE_SPECS[venue]
    files = {name: f"official:{name}\n".encode() for name in spec["files"]}
    for name in spec.get("ignored_files", ()):
        files[name] = b"ignored-pdf"
    if venue == "neurips":
        files["neurips_2026.tex"] = (
            b"\\documentclass{article}\n\\usepackage{neurips_2026}\n"
        )
    else:
        files["example_paper.tex"] = (
            b"\\documentclass{article}\n\\usepackage{icml2026}\n"
        )
    return files


def _fixture_spec(venue: str, payload: bytes) -> dict:
    spec = dict(resources._OFFICIAL_TEMPLATE_SPECS[venue])
    spec["sha256"] = hashlib.sha256(payload).hexdigest()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        spec["file_sha256"] = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in spec["files"]
        }
    return spec


class OfficialVenueTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._cache_directory = tempfile.TemporaryDirectory()
        self._cache_environment = mock.patch.dict(
            os.environ,
            {"XDG_CACHE_HOME": self._cache_directory.name},
        )
        self._cache_environment.start()

    def tearDown(self) -> None:
        self._cache_environment.stop()
        self._cache_directory.cleanup()

    def test_neurips_materialization_is_verified_and_auditable(self) -> None:
        payload = _zip_payload(_fixture_files("neurips"))
        response = _MemoryResponse(payload, content_length=len(payload))

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "latex"
            with (
                mock.patch.dict(
                    resources._OFFICIAL_TEMPLATE_SPECS,
                    {"neurips": _fixture_spec("neurips", payload)},
                ),
                mock.patch(
                    "ai_scientist.resources.urlopen", return_value=response
                ) as fetch,
            ):
                result = materialize_latex_template("NeurIPS", destination)

            self.assertEqual(result, destination)
            self.assertEqual(
                (destination / "template.tex").read_bytes(),
                (destination / "neurips_2026.tex").read_bytes(),
            )
            receipt = json.loads(
                (destination / "template_source.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["schema"], "xscientist.template-source.v1")
            self.assertEqual(receipt["venue"], "neurips")
            self.assertEqual(receipt["year"], 2026)
            self.assertEqual(receipt["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertTrue(receipt["url"].startswith("https://media.neurips.cc/"))
            self.assertTrue(receipt["verified_at"].endswith("Z"))
            self.assertEqual(
                set(receipt["files"]),
                {
                    "checklist.tex",
                    "neurips_2026.sty",
                    "neurips_2026.tex",
                    "template.tex",
                },
            )
            self.assertEqual(
                set(receipt["source_file_hashes"]),
                set(resources._OFFICIAL_TEMPLATE_SPECS["neurips"]["files"]),
            )
            with mock.patch.dict(
                resources._OFFICIAL_TEMPLATE_SPECS,
                {"neurips": _fixture_spec("neurips", payload)},
            ):
                self.assertTrue(
                    verify_latex_template_source("neurips", destination)["ok"]
                )
            request = fetch.call_args.args[0]
            self.assertEqual(
                request.full_url,
                resources._OFFICIAL_TEMPLATE_SPECS["neurips"]["url"],
            )

    def test_icml_materialization_excludes_archive_pdfs(self) -> None:
        payload = _zip_payload(_fixture_files("icml"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "latex"
            with (
                mock.patch.dict(
                    resources._OFFICIAL_TEMPLATE_SPECS,
                    {"icml": _fixture_spec("icml", payload)},
                ),
                mock.patch(
                    "ai_scientist.resources.urlopen",
                    return_value=_MemoryResponse(payload),
                ),
            ):
                materialize_latex_template("icml", destination)

            self.assertEqual(
                (destination / "template.tex").read_bytes(),
                (destination / "example_paper.tex").read_bytes(),
            )
            self.assertFalse(list(destination.glob("*.pdf")))
            receipt = json.loads(
                (destination / "template_source.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("example_paper.pdf", receipt["files"])
            self.assertNotIn("icml_numpapers.pdf", receipt["files"])
            with mock.patch.dict(
                resources._OFFICIAL_TEMPLATE_SPECS,
                {"icml": _fixture_spec("icml", payload)},
            ):
                self.assertTrue(verify_latex_template_source("icml", destination)["ok"])

    def test_source_or_final_style_tampering_blocks_template_attestation(self) -> None:
        payload = _zip_payload(_fixture_files("neurips"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "latex"
            with (
                mock.patch.dict(
                    resources._OFFICIAL_TEMPLATE_SPECS,
                    {"neurips": _fixture_spec("neurips", payload)},
                ),
                mock.patch(
                    "ai_scientist.resources.urlopen",
                    return_value=_MemoryResponse(payload),
                ),
            ):
                materialize_latex_template("neurips", destination)
                (destination / "neurips_2026.sty").write_text(
                    "tampered", encoding="utf-8"
                )
                source_report = verify_latex_template_source("neurips", destination)
                (destination / "neurips_2026.sty").write_bytes(
                    _fixture_files("neurips")["neurips_2026.sty"]
                )
                (destination / "template.tex").write_text(
                    "\\documentclass{article}\n", encoding="utf-8"
                )
                style_report = verify_latex_template_source("neurips", destination)

            self.assertFalse(source_report["ok"])
            self.assertIn(
                "official_template_source_file_hash_mismatch",
                source_report["errors"],
            )
            self.assertFalse(style_report["ok"])
            self.assertIn(
                "official_template_manuscript_style_mismatch",
                style_report["errors"],
            )

    def test_oversized_source_file_fails_closed_without_unbounded_read(self) -> None:
        payload = _zip_payload(_fixture_files("neurips"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "latex"
            with (
                mock.patch.dict(
                    resources._OFFICIAL_TEMPLATE_SPECS,
                    {"neurips": _fixture_spec("neurips", payload)},
                ),
                mock.patch(
                    "ai_scientist.resources.urlopen",
                    return_value=_MemoryResponse(payload),
                ),
            ):
                materialize_latex_template("neurips", destination)
                source_path = destination / "neurips_2026.sty"
                with source_path.open("wb") as source:
                    source.truncate(resources._MAX_TEMPLATE_SOURCE_FILE_BYTES + 1)
                report = verify_latex_template_source("neurips", destination)

            self.assertFalse(report["ok"])
            self.assertIn(
                "official_template_source_file_unreadable",
                report["errors"],
            )

    def test_style_attestation_requires_active_nonconflicting_declaration(self) -> None:
        payload = _zip_payload(_fixture_files("neurips"))
        fixture_spec = _fixture_spec("neurips", payload)
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "latex"
            with (
                mock.patch.dict(
                    resources._OFFICIAL_TEMPLATE_SPECS,
                    {"neurips": fixture_spec},
                ),
                mock.patch(
                    "ai_scientist.resources.urlopen",
                    return_value=_MemoryResponse(payload),
                ),
            ):
                materialize_latex_template("neurips", destination)

                (destination / "template.tex").write_text(
                    "\\documentclass{article}\n"
                    "\\usepackage{geometry}\n"
                    "% \\usepackage{neurips_2026}\n",
                    encoding="utf-8",
                )
                commented = verify_latex_template_source("neurips", destination)

                (destination / "template.tex").write_text(
                    "\\documentclass{article}\n"
                    "\\newcommand{\\percent}{\\%}\n"
                    "\\usepackage{neurips_2026}\n",
                    encoding="utf-8",
                )
                escaped_percent = verify_latex_template_source("neurips", destination)

                (destination / "template.tex").write_text(
                    "\\documentclass{article}\n"
                    "\\usepackage{neurips_2026}\n"
                    "\\usepackage{icml2026}\n",
                    encoding="utf-8",
                )
                conflicting = verify_latex_template_source("neurips", destination)

        self.assertFalse(commented["ok"])
        self.assertIn(
            "official_template_manuscript_style_mismatch",
            commented["errors"],
        )
        self.assertTrue(escaped_percent["ok"], escaped_percent)
        self.assertFalse(conflicting["ok"])
        self.assertIn(
            "official_template_manuscript_conflicting_style",
            conflicting["errors"],
        )
        stripped = resources._strip_latex_comments("\\% visible % hidden declaration\n")
        self.assertIn("\\% visible", stripped)
        self.assertNotIn("hidden declaration", stripped)

    def test_forged_receipt_cannot_reauthorize_tampered_official_source(self) -> None:
        payload = _zip_payload(_fixture_files("neurips"))
        fixture_spec = _fixture_spec("neurips", payload)
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "latex"
            with (
                mock.patch.dict(
                    resources._OFFICIAL_TEMPLATE_SPECS,
                    {"neurips": fixture_spec},
                ),
                mock.patch(
                    "ai_scientist.resources.urlopen",
                    return_value=_MemoryResponse(payload),
                ),
            ):
                materialize_latex_template("neurips", destination)
                tampered = b"% forged style\n"
                (destination / "neurips_2026.sty").write_bytes(tampered)
                receipt_path = destination / "template_source.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt["source_file_hashes"]["neurips_2026.sty"] = (
                    "sha256:" + hashlib.sha256(tampered).hexdigest()
                )
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                report = verify_latex_template_source("neurips", destination)

            self.assertFalse(report["ok"])
            self.assertIn(
                "official_template_source_hashes_mismatch",
                report["errors"],
            )
            self.assertIn(
                "official_template_source_file_hash_mismatch",
                report["errors"],
            )

    def test_verified_archive_cache_supports_offline_reuse(self) -> None:
        payload = _zip_payload(_fixture_files("neurips"))
        fixture_spec = _fixture_spec("neurips", payload)
        with tempfile.TemporaryDirectory() as temporary_directory:
            first_destination = Path(temporary_directory) / "first"
            second_destination = Path(temporary_directory) / "second"
            cache_path = resources._template_archive_cache_path(
                "neurips",
                fixture_spec,
            )
            cache_path.parent.mkdir(parents=True)
            cache_path.write_bytes(b"corrupt-cache-entry")
            fetch = mock.Mock(return_value=_MemoryResponse(payload))
            with (
                mock.patch.dict(
                    resources._OFFICIAL_TEMPLATE_SPECS,
                    {"neurips": fixture_spec},
                ),
                mock.patch("ai_scientist.resources.urlopen", fetch),
            ):
                materialize_latex_template("neurips", first_destination)
                materialize_latex_template("neurips", second_destination)

            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(
                (first_destination / "template.tex").read_bytes(),
                (second_destination / "template.tex").read_bytes(),
            )
            self.assertEqual(cache_path.read_bytes(), payload)

    def test_hash_mismatch_fails_without_partial_destination(self) -> None:
        payload = b"not-the-pinned-official-archive"
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "latex"
            with mock.patch(
                "ai_scientist.resources.urlopen",
                return_value=_MemoryResponse(payload, content_length=len(payload)),
            ):
                with self.assertRaisesRegex(OfficialTemplateError, "SHA-256"):
                    materialize_latex_template("neurips", destination)
            self.assertFalse(destination.exists())

    def test_download_is_bounded_by_header_and_actual_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            first_destination = Path(temporary_directory) / "header-limit"
            with (
                mock.patch.object(resources, "_MAX_TEMPLATE_ARCHIVE_BYTES", 8),
                mock.patch(
                    "ai_scientist.resources.urlopen",
                    return_value=_MemoryResponse(b"small", content_length=9),
                ),
            ):
                with self.assertRaisesRegex(OfficialTemplateError, "maximum allowed"):
                    materialize_latex_template("neurips", first_destination)

            second_destination = Path(temporary_directory) / "read-limit"
            with (
                mock.patch.object(resources, "_MAX_TEMPLATE_ARCHIVE_BYTES", 8),
                mock.patch(
                    "ai_scientist.resources.urlopen",
                    return_value=_MemoryResponse(b"123456789"),
                ),
            ):
                with self.assertRaisesRegex(OfficialTemplateError, "maximum allowed"):
                    materialize_latex_template("neurips", second_destination)

            self.assertFalse(first_destination.exists())
            self.assertFalse(second_destination.exists())

    def test_archive_rejects_path_traversal_and_unexpected_files(self) -> None:
        for unsafe_name in ("../escape.tex", "unapproved.txt"):
            with self.subTest(unsafe_name=unsafe_name):
                files = _fixture_files("neurips")
                files[unsafe_name] = b"unsafe"
                payload = _zip_payload(files)
                with tempfile.TemporaryDirectory() as temporary_directory:
                    destination = Path(temporary_directory) / "latex"
                    with (
                        mock.patch.dict(
                            resources._OFFICIAL_TEMPLATE_SPECS,
                            {"neurips": _fixture_spec("neurips", payload)},
                        ),
                        mock.patch(
                            "ai_scientist.resources.urlopen",
                            return_value=_MemoryResponse(payload),
                        ),
                    ):
                        with self.assertRaises(OfficialTemplateError):
                            materialize_latex_template("neurips", destination)
                    self.assertFalse(destination.exists())

    def test_archive_rejects_symbolic_links(self) -> None:
        files = _fixture_files("neurips")
        files["checklist.tex"] = b"../outside"
        payload = _zip_payload(files, symlinks={"checklist.tex"})
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "latex"
            with (
                mock.patch.dict(
                    resources._OFFICIAL_TEMPLATE_SPECS,
                    {"neurips": _fixture_spec("neurips", payload)},
                ),
                mock.patch(
                    "ai_scientist.resources.urlopen",
                    return_value=_MemoryResponse(payload),
                ),
            ):
                with self.assertRaisesRegex(OfficialTemplateError, "symbolic link"):
                    materialize_latex_template("neurips", destination)
            self.assertFalse(destination.exists())

    def test_top_venue_page_limits_override_generic_writeup_type(self) -> None:
        self.assertEqual(resolve_page_limit("normal"), 8)
        self.assertEqual(resolve_page_limit("normal", "neurips"), 9)
        self.assertEqual(resolve_page_limit("journal", "icml"), 8)
        neurips_plan = build_writeup_execution_plan(
            "normal",
            num_cite_rounds=1,
            writeup_retries=1,
            target_venue="neurips",
        )
        icml_plan = build_writeup_execution_plan(
            "normal",
            num_cite_rounds=1,
            writeup_retries=1,
            target_venue="icml",
        )
        self.assertEqual(neurips_plan["page_limit"], 9)
        self.assertEqual(icml_plan["page_limit"], 8)

    def test_official_examples_are_rebound_to_clean_external_bibliography(
        self,
    ) -> None:
        from ai_scientist.perform_writeup import (
            _latex_with_bound_bibliography,
            _prepare_bibliography_binding,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            latex_folder = Path(temporary_directory)
            template = latex_folder / "template.tex"
            template.write_text(
                "\\documentclass{article}\n"
                "\\begin{document}\n"
                "\\nocite{langley00}\n"
                "\\bibliography{example_paper}\n"
                "\\bibliographystyle{icml2026}\n"
                "\\end{document}\n",
                encoding="utf-8",
            )
            (latex_folder / "example_paper.bib").write_text(
                "@article{langley00, title={Unrelated sample}}\n",
                encoding="utf-8",
            )

            binding = _prepare_bibliography_binding(
                str(latex_folder),
                str(template),
                "icml",
            )

            rebound = template.read_text(encoding="utf-8")
            self.assertEqual(binding["mode"], "external")
            self.assertEqual(Path(binding["path"]).read_text(encoding="utf-8"), "")
            self.assertIn("\\bibliography{references}", rebound)
            self.assertNotIn("\\nocite{langley00}", rebound)
            self.assertNotIn("Unrelated sample", rebound)

            Path(binding["path"]).write_text(
                "@article{verified, title={Verified source}}\n",
                encoding="utf-8",
            )
            guardrail_input = _latex_with_bound_bibliography(
                rebound + "\\cite{verified}\n",
                binding,
            )
            self.assertIn("@article{verified", guardrail_input)

    def test_neurips_sample_reference_block_is_replaced_by_bibtex_binding(
        self,
    ) -> None:
        from ai_scientist.perform_writeup import _prepare_bibliography_binding

        with tempfile.TemporaryDirectory() as temporary_directory:
            latex_folder = Path(temporary_directory)
            template = latex_folder / "template.tex"
            template.write_text(
                "\\documentclass{article}\n"
                "\\begin{document}\n"
                "\\section*{References}\n"
                "[1] Official formatting example only.\n"
                "\\appendix\n"
                "\\end{document}\n",
                encoding="utf-8",
            )

            binding = _prepare_bibliography_binding(
                str(latex_folder),
                str(template),
                "neurips",
            )

            rebound = template.read_text(encoding="utf-8")
            self.assertEqual(binding["mode"], "external")
            self.assertNotIn("Official formatting example only", rebound)
            self.assertIn("\\bibliographystyle{plainnat}", rebound)
            self.assertIn("\\bibliography{references}", rebound)

    def test_writeup_fails_closed_when_official_template_verification_fails(
        self,
    ) -> None:
        from ai_scientist.perform_writeup import perform_writeup

        output = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch(
                "ai_scientist.perform_writeup.materialize_latex_template",
                side_effect=OfficialTemplateError("pinned hash mismatch"),
            ),
            mock.patch("ai_scientist.perform_writeup.latex_template_dir") as legacy,
        ):
            with redirect_stdout(output):
                result = perform_writeup(
                    temporary_directory,
                    no_writing=True,
                    target_venue="neurips",
                )

        self.assertFalse(result)
        legacy.assert_not_called()
        self.assertIn("Official venue template verification failed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
