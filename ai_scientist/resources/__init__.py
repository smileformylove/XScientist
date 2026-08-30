from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

from ai_scientist.utils.safe_files import BoundedFileError, read_bounded_regular_file

_CONFIG_NAMES = {
    "default": "bfts_default.yaml",
    "deep": "bfts_deep.yaml",
    "glm53": "bfts_glm53.yaml",
}

_LATEX_TEMPLATE_NAMES = {
    "icbinb": "blank_icbinb_latex",
    "icml": "blank_icml_latex",
    "normal": "blank_icml_latex",
    "journal": "blank_icml_latex",
    "extended": "blank_icbinb_latex",
}

_MAX_TEMPLATE_ARCHIVE_BYTES = 16 * 1024 * 1024
_MAX_TEMPLATE_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
_MAX_TEMPLATE_SOURCE_FILE_BYTES = 16 * 1024 * 1024
_TEMPLATE_DOWNLOAD_TIMEOUT_SECONDS = 30

_OFFICIAL_TEMPLATE_SPECS = {
    "neurips": {
        "year": 2026,
        "url": "https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip",
        "sha256": "82473931e3ef710fcd3f4a8cd4119b9de32e56825f90f9e5a6d55f2d01b817d9",
        "files": frozenset({"neurips_2026.tex", "checklist.tex", "neurips_2026.sty"}),
        "file_sha256": {
            "checklist.tex": "780ba13c480f652dcc42e69ed61a752ce0ea270f15d332d4a45b059dabad84f6",
            "neurips_2026.sty": "c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a",
            "neurips_2026.tex": "cf4cee7991665306d1daaa3985be4feec7f8889d6d072ffa12f99a8e1537d797",
        },
        "ignored_files": frozenset(),
        "template_file": "neurips_2026.tex",
    },
    "icml": {
        "year": 2026,
        "url": "https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip",
        "sha256": "8b29290f5828e176debb57ea9cc00252502973d55ea561a2f18a7f0a326bfc6c",
        "files": frozenset(
            {
                "algorithm.sty",
                "algorithmic.sty",
                "example_paper.bib",
                "example_paper.tex",
                "fancyhdr.sty",
                "icml2026.bst",
                "icml2026.sty",
            }
        ),
        "file_sha256": {
            "algorithm.sty": "93fd0eb31c112eb405833db8f1d7f5d238c7e691b1c05680d7276e68f36d564a",
            "algorithmic.sty": "48d18794a5d97c0479a588cc2eac0917992feb9da83acc4631b8f55757d80f9b",
            "example_paper.bib": "df950103d38f9cfc81b1f40d84c9be2a3525d046d2991a6973a4446922c06bd1",
            "example_paper.tex": "c2ca8140bf255d1ff77d1278eb3eefed4018b4b1e71065b8e1ccbc68b74c8acf",
            "fancyhdr.sty": "9130c52f91087abc6d223164ffa587e207e3257fcbcd069ef09ecb5391043f14",
            "icml2026.bst": "0ec3d5eb9b02efb7e0b44a32f3775882f42a743d0bdc618f34e6936309b98764",
            "icml2026.sty": "7cdcf90f6a59c5219e7f15c88f7ed09fcaf598dad91e6cdddc4dc3cb0e397a95",
        },
        # These files are present in the official archive but are not required
        # to compile a manuscript and must not be copied into a project.
        "ignored_files": frozenset({"example_paper.pdf", "icml_numpapers.pdf"}),
        "template_file": "example_paper.tex",
    },
}


class OfficialTemplateError(RuntimeError):
    """Raised when an official venue template cannot be verified safely."""


_USEPACKAGE_DECLARATION = re.compile(
    r"^[ \t]*\\usepackage(?:\[[^\]]*\])?\s*\{([^{}]+)\}",
    re.MULTILINE,
)
_VENUE_STYLE_PACKAGE = re.compile(r"(?:neurips_?\d{4}|icml\d{4})")


def _strip_latex_comments(source: str) -> str:
    """Remove TeX comments while preserving percent signs escaped by ``\\``."""

    uncommented: list[str] = []
    for line in source.splitlines(keepends=True):
        for index, character in enumerate(line):
            if character != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                newline = ""
                if line.endswith("\r\n"):
                    newline = "\r\n"
                elif line.endswith(("\r", "\n")):
                    newline = line[-1]
                line = line[:index] + newline
                break
        uncommented.append(line)
    return "".join(uncommented)


def _active_latex_packages(source: str) -> set[str]:
    uncommented = _strip_latex_comments(source)
    return {
        package.strip()
        for declaration in _USEPACKAGE_DECLARATION.findall(uncommented)
        for package in declaration.split(",")
        if package.strip()
    }


def _read_bounded_response(response, *, maximum_bytes: int) -> bytes:
    content_length = response.headers.get("Content-Length")
    declared_length = None
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except (TypeError, ValueError) as exc:
            raise OfficialTemplateError(
                "Official template response has an invalid Content-Length header."
            ) from exc
        if declared_length < 0 or declared_length > maximum_bytes:
            raise OfficialTemplateError(
                "Official template archive exceeds the maximum allowed download size "
                f"of {maximum_bytes} bytes."
            )

    payload = response.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise OfficialTemplateError(
            "Official template archive exceeds the maximum allowed download size "
            f"of {maximum_bytes} bytes."
        )
    if declared_length is not None and declared_length != len(payload):
        raise OfficialTemplateError(
            "Official template response length does not match Content-Length."
        )
    return payload


def _download_official_template(spec: dict) -> bytes:
    request = Request(
        str(spec["url"]),
        headers={
            "Accept": "application/zip, application/octet-stream",
            "Accept-Encoding": "identity",
            "User-Agent": "XScientist/official-template-fetcher",
        },
    )
    try:
        with urlopen(request, timeout=_TEMPLATE_DOWNLOAD_TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", None)
            if status is not None and not 200 <= int(status) < 300:
                raise OfficialTemplateError(
                    f"Official template server returned HTTP status {status}."
                )
            payload = _read_bounded_response(
                response,
                maximum_bytes=_MAX_TEMPLATE_ARCHIVE_BYTES,
            )
    except OfficialTemplateError:
        raise
    except Exception as exc:
        raise OfficialTemplateError(
            "Could not download the pinned official template. "
            f"URL: {spec['url']}; expected SHA-256: {spec['sha256']}. "
            "Check network access and retry; XScientist will not fall back to a "
            f"different venue or year. Underlying error: {exc}"
        ) from exc

    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != spec["sha256"]:
        raise OfficialTemplateError(
            "Official template archive failed SHA-256 verification: "
            f"expected {spec['sha256']}, got {actual_sha256}."
        )
    return payload


def _template_archive_cache_path(venue: str, spec: dict) -> Path:
    xdg_cache_home = str(os.environ.get("XDG_CACHE_HOME") or "").strip()
    cache_home = (
        Path(xdg_cache_home).expanduser() if xdg_cache_home else Path.home() / ".cache"
    )
    return (
        cache_home
        / "xscientist"
        / "official-templates"
        / f"{venue}-{spec['year']}-{spec['sha256']}.zip"
    )


def _read_verified_template_cache(
    cache_path: Path, expected_sha256: str
) -> bytes | None:
    try:
        with cache_path.open("rb") as cached_archive:
            payload = cached_archive.read(_MAX_TEMPLATE_ARCHIVE_BYTES + 1)
    except OSError:
        return None
    if len(payload) > _MAX_TEMPLATE_ARCHIVE_BYTES:
        return None
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        return None
    return payload


def _store_verified_template_cache(cache_path: Path, payload: bytes) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(
            prefix=".template-cache-",
            dir=cache_path.parent,
        ) as temporary_directory:
            staged = Path(temporary_directory) / cache_path.name
            staged.write_bytes(payload)
            staged.chmod(0o600)
            staged.replace(cache_path)
    except OSError:
        # A read-only or unavailable cache must not block a verified download.
        return


def _load_official_template(venue: str, spec: dict) -> bytes:
    cache_path = _template_archive_cache_path(venue, spec)
    cached_payload = _read_verified_template_cache(cache_path, str(spec["sha256"]))
    if cached_payload is not None:
        return cached_payload

    payload = _download_official_template(spec)
    _store_verified_template_cache(cache_path, payload)
    return payload


def _validated_template_files(payload: bytes, spec: dict) -> dict[str, bytes]:
    expected_files = set(spec["files"])
    expected_hashes = spec.get("file_sha256")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != expected_files:
        raise OfficialTemplateError(
            "Official template specification is missing pinned source-file hashes."
        )
    ignored_files = set(spec.get("ignored_files", ()))
    allowed_archive_files = expected_files | ignored_files
    extracted: dict[str, bytes] = {}
    seen: set[str] = set()
    total_uncompressed_bytes = 0

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.infolist():
                name = member.filename
                path = PurePosixPath(name)
                if (
                    not name
                    or "\\" in name
                    or path.is_absolute()
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or len(path.parts) != 1
                ):
                    raise OfficialTemplateError(
                        f"Official template archive contains an unsafe path: {name!r}."
                    )
                if member.is_dir():
                    raise OfficialTemplateError(
                        f"Official template archive contains an unexpected directory: {name!r}."
                    )
                unix_mode = member.external_attr >> 16
                if stat.S_ISLNK(unix_mode):
                    raise OfficialTemplateError(
                        f"Official template archive contains a symbolic link: {name!r}."
                    )
                if member.flag_bits & 0x1:
                    raise OfficialTemplateError(
                        f"Official template archive contains an encrypted file: {name!r}."
                    )
                if name in seen:
                    raise OfficialTemplateError(
                        f"Official template archive contains a duplicate file: {name!r}."
                    )
                seen.add(name)
                if name not in allowed_archive_files:
                    raise OfficialTemplateError(
                        f"Official template archive contains an unexpected file: {name!r}."
                    )

                total_uncompressed_bytes += member.file_size
                if total_uncompressed_bytes > _MAX_TEMPLATE_UNCOMPRESSED_BYTES:
                    raise OfficialTemplateError(
                        "Official template archive exceeds the maximum allowed "
                        "uncompressed size."
                    )
                if name in expected_files:
                    extracted[name] = archive.read(member)
    except OfficialTemplateError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise OfficialTemplateError(
            f"Official template archive is not a valid readable ZIP file: {exc}"
        ) from exc

    missing = expected_files - extracted.keys()
    if missing:
        raise OfficialTemplateError(
            "Official template archive is missing required files: "
            + ", ".join(sorted(missing))
        )
    mismatched = [
        name
        for name, content in extracted.items()
        if hashlib.sha256(content).hexdigest() != expected_hashes.get(name)
    ]
    if mismatched:
        raise OfficialTemplateError(
            "Official template source file failed pinned SHA-256 verification: "
            + ", ".join(sorted(mismatched))
        )
    return extracted


def materialize_latex_template(
    target_venue: str,
    destination: str | Path,
) -> Path:
    """Download, verify, and atomically materialize a current official template.

    Only explicitly pinned NeurIPS and ICML template archives are supported. The
    function intentionally has no packaged-template fallback: a network, hash,
    or archive validation failure must block a venue-targeted write-up.
    """

    normalized = str(target_venue or "").strip().lower()
    try:
        spec = _OFFICIAL_TEMPLATE_SPECS[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(_OFFICIAL_TEMPLATE_SPECS))
        raise ValueError(
            f"No pinned official template for {target_venue!r}; expected: {choices}."
        ) from exc

    destination_path = Path(destination).expanduser()
    if destination_path.exists():
        raise FileExistsError(
            f"Official template destination already exists: {destination_path}"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _load_official_template(normalized, spec)
    extracted = _validated_template_files(payload, spec)
    output_files = sorted(set(extracted) | {"template.tex"})

    with tempfile.TemporaryDirectory(
        prefix=f".{destination_path.name}.official-",
        dir=destination_path.parent,
    ) as temporary_parent:
        staged = Path(temporary_parent) / destination_path.name
        staged.mkdir()
        for name, content in extracted.items():
            (staged / name).write_bytes(content)
        shutil.copyfile(staged / str(spec["template_file"]), staged / "template.tex")
        receipt = {
            "schema": "xscientist.template-source.v1",
            "venue": normalized,
            "year": spec["year"],
            "url": spec["url"],
            "sha256": spec["sha256"],
            "files": output_files,
            "source_file_hashes": {
                name: "sha256:" + str(spec["file_sha256"][name])
                for name in sorted(extracted)
            },
            "verified_at": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        }
        (staged / "template_source.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged.replace(destination_path)
    return destination_path


def verify_latex_template_source(
    target_venue: str,
    destination: str | Path,
) -> dict[str, object]:
    """Revalidate official source bytes and the final manuscript style binding."""

    normalized = str(target_venue or "").strip().lower()
    spec = _OFFICIAL_TEMPLATE_SPECS.get(normalized)
    errors: list[str] = []
    destination_path = Path(destination).expanduser().resolve()
    if spec is None:
        return {
            "ok": False,
            "status": "blocked",
            "errors": ["official_template_target_unsupported"],
            "venue": normalized or None,
        }

    receipt_path = destination_path / "template_source.json"
    try:
        receipt_payload = read_bounded_regular_file(
            receipt_path,
            maximum=1024 * 1024,
            label="official_template_receipt",
        )
        receipt = json.loads(receipt_payload.decode("utf-8"))
    except (BoundedFileError, UnicodeError, ValueError, TypeError):
        return {
            "ok": False,
            "status": "blocked",
            "errors": ["official_template_receipt_unreadable"],
            "venue": normalized,
        }
    if not isinstance(receipt, dict):
        return {
            "ok": False,
            "status": "blocked",
            "errors": ["official_template_receipt_not_object"],
            "venue": normalized,
        }

    expected_files = sorted(set(spec["files"]) | {"template.tex"})
    if receipt.get("schema") != "xscientist.template-source.v1":
        errors.append("official_template_receipt_schema_invalid")
    for field in ("venue", "year", "url", "sha256"):
        expected = normalized if field == "venue" else spec[field]
        if receipt.get(field) != expected:
            errors.append(f"official_template_receipt_{field}_mismatch")
    if receipt.get("files") != expected_files:
        errors.append("official_template_receipt_file_set_mismatch")

    expected_source_hashes = {
        name: "sha256:" + str(spec["file_sha256"][name])
        for name in sorted(spec["files"])
    }
    source_hashes = receipt.get("source_file_hashes")
    if source_hashes != expected_source_hashes:
        errors.append("official_template_source_hashes_mismatch")
    for name in sorted(spec["files"]):
        source_path = destination_path / name
        expected_hash = expected_source_hashes[name]
        try:
            source_payload = read_bounded_regular_file(
                source_path,
                maximum=_MAX_TEMPLATE_SOURCE_FILE_BYTES,
                label="official_template_source_file",
            )
            actual_hash = "sha256:" + hashlib.sha256(source_payload).hexdigest()
        except BoundedFileError as exc:
            errors.append(
                "official_template_source_file_missing"
                if exc.reason in {"missing", "not_regular", "symlink_rejected"}
                else "official_template_source_file_unreadable"
            )
            continue
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            errors.append("official_template_source_file_hash_mismatch")

    manuscript_path = destination_path / "template.tex"
    try:
        manuscript_payload = read_bounded_regular_file(
            manuscript_path,
            maximum=16 * 1024 * 1024,
            label="official_template_manuscript",
        )
        manuscript = manuscript_payload.decode("utf-8")
    except (BoundedFileError, UnicodeError):
        manuscript = ""
        errors.append("official_template_manuscript_unreadable")
    required_style = "neurips_2026" if normalized == "neurips" else "icml2026"
    active_packages = _active_latex_packages(manuscript)
    if required_style not in active_packages:
        errors.append("official_template_manuscript_style_mismatch")
    conflicting_styles = {
        package
        for package in active_packages
        if package != required_style and _VENUE_STYLE_PACKAGE.fullmatch(package)
    }
    if conflicting_styles:
        errors.append("official_template_manuscript_conflicting_style")

    return {
        "ok": not errors,
        "status": "ready" if not errors else "blocked",
        "errors": sorted(set(errors)),
        "venue": normalized,
        "year": spec["year"],
        "archive_sha256": spec["sha256"],
    }


def bfts_config_path(profile: str = "default") -> Path:
    """Return the installed BFTS configuration for a supported profile."""

    normalized = str(profile or "default").strip().lower()
    try:
        filename = _CONFIG_NAMES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(_CONFIG_NAMES))
        raise ValueError(
            f"Unknown BFTS config profile {profile!r}; expected one of: {choices}"
        ) from exc
    resource = files("ai_scientist.resources.configs").joinpath(filename)
    path = Path(str(resource))
    if not path.is_file():
        raise FileNotFoundError(f"Packaged BFTS config is missing: {path}")
    return path


def resolve_bfts_config_path(
    value: str | Path | None = None, *, base_dir: str | Path | None = None
) -> Path:
    """Resolve an explicit path, source config, or packaged profile."""

    if value is None:
        return bfts_config_path("default")
    text = str(value).strip()
    profile_aliases = {
        "default": "default",
        "bfts_config.yaml": "default",
        "deep": "deep",
        "bfts_config_deep.yaml": "deep",
        "glm53": "glm53",
        "glm-5.3": "glm53",
        "bfts_glm53.yaml": "glm53",
        "bfts_config_glm53.yaml": "glm53",
    }
    candidate = Path(text).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        relative_candidate = Path(base_dir or Path.cwd()).expanduser() / candidate
        if relative_candidate.is_file():
            return relative_candidate.resolve()
        source_root = Path(__file__).resolve().parents[2]
        if (source_root / "pyproject.toml").is_file():
            source_candidate = source_root / "configs" / "bfts" / candidate.name
            if source_candidate.is_file():
                return source_candidate.resolve()
    profile = profile_aliases.get(text.lower()) or profile_aliases.get(
        candidate.name.lower()
    )
    if profile is not None:
        return bfts_config_path(profile)
    choices = ", ".join(repr(name) for name in sorted(_CONFIG_NAMES))
    raise FileNotFoundError(
        f"BFTS config not found: {value}. Use an existing YAML path or one of "
        f"the packaged profiles: {choices}."
    )


def package_root() -> Path:
    """Return the installed ``ai_scientist`` package directory."""

    return Path(__file__).resolve().parent.parent


def latex_template_dir(template: str) -> Path:
    """Return a packaged LaTeX template directory."""

    normalized = str(template or "").strip().lower()
    directory_name = _LATEX_TEMPLATE_NAMES.get(normalized, template)
    path = package_root() / str(directory_name)
    if not path.is_dir():
        choices = ", ".join(sorted(_LATEX_TEMPLATE_NAMES))
        raise ValueError(
            f"Unknown LaTeX template {template!r}; expected one of: {choices}"
        )
    return path


def idea_resource_path(filename: str = "i_cant_believe_its_not_better.json") -> Path:
    """Return a packaged example idea or workshop-description file."""

    name = Path(filename).name
    path = package_root() / "ideas" / name
    if not path.is_file():
        raise FileNotFoundError(f"Packaged idea resource is missing: {name}")
    return path


__all__ = [
    "OfficialTemplateError",
    "bfts_config_path",
    "idea_resource_path",
    "latex_template_dir",
    "materialize_latex_template",
    "package_root",
    "resolve_bfts_config_path",
    "verify_latex_template_source",
]
