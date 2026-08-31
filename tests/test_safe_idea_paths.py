from __future__ import annotations

from pathlib import Path

import pytest

from ai_scientist.config.paths import (
    confined_path,
    content_identity,
    get_batch_dir,
    get_experiment_dir,
    get_idea_path,
    get_paper_dir,
    get_project_dir,
    idea_storage_key,
    safe_path_component,
)


def test_untrusted_idea_names_are_labels_not_paths(tmp_path: Path) -> None:
    idea = {"Name": "../../../../escaped", "Title": "Traversal attempt"}

    key = idea_storage_key(idea, idea_index=7)
    experiment = get_experiment_dir(
        idea["Name"],
        0,
        output_root=tmp_path,
        idea_identity=idea,
        idea_index=7,
    )

    assert key.startswith("0007-escaped-")
    assert "/" not in key and "\\" not in key and ".." not in key
    assert experiment.is_relative_to((tmp_path / "experiments").resolve())
    assert tmp_path.parent / "escaped" != experiment


def test_content_identity_separates_duplicate_labels_and_is_stable() -> None:
    first = {"Name": "same_name", "Title": "First"}
    second = {"Name": "same_name", "Title": "Second"}

    assert idea_storage_key(first) == idea_storage_key(dict(first))
    assert idea_storage_key(first) != idea_storage_key(second)


def test_storage_key_bounds_unicode_and_oversized_names() -> None:
    key = idea_storage_key({"Name": "研究/" + "A" * 500})

    slug, digest = key.rsplit("-", 1)
    assert len(slug) <= 48
    assert len(digest) == 12
    assert safe_path_component("研究").startswith("item-")
    assert safe_path_component("研究") != safe_path_component("科学")
    assert safe_path_component("CON.txt").startswith("item-")


def test_experiment_paths_do_not_collide_within_one_process(tmp_path: Path) -> None:
    idea = {"Name": "same_name", "Title": "Same content"}

    first = get_experiment_dir(
        idea["Name"], output_root=tmp_path, idea_identity=idea, idea_index=0
    )
    second = get_experiment_dir(
        idea["Name"], output_root=tmp_path, idea_identity=idea, idea_index=0
    )

    assert first != second


def test_confined_path_rejects_traversal_and_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()

    with pytest.raises(ValueError, match="escapes"):
        confined_path(root, "..", "outside", "artifact.json")

    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="escapes"):
        confined_path(root, "linked", "artifact.json")


def test_other_named_artifacts_stay_within_their_roots(tmp_path: Path) -> None:
    idea_path = get_idea_path("../../idea\\escape", output_root=tmp_path)
    paper_path = get_paper_dir(
        "../../paper\\escape",
        paper_type="../journal",
        timestamp="../../20260831",
        output_root=tmp_path,
    )
    batch_path = get_batch_dir("../../batch\\escape", output_root=tmp_path)

    assert idea_path.is_relative_to((tmp_path / "ideas").resolve())
    assert paper_path.is_relative_to((tmp_path / "papers").resolve())
    assert batch_path.is_relative_to((tmp_path / "batches").resolve())
    for path in (idea_path, paper_path, batch_path):
        assert ".." not in path.name
        assert "/" not in path.name and "\\" not in path.name


def test_lossy_labels_receive_distinct_stable_artifact_keys(tmp_path: Path) -> None:
    assert get_idea_path("Alpha+Beta", output_root=tmp_path) != get_idea_path(
        "Alpha Beta", output_root=tmp_path
    )
    assert get_batch_dir("Run:One", output_root=tmp_path) != get_batch_dir(
        "Run?One", output_root=tmp_path
    )
    assert get_idea_path("A" * 120 + "x", output_root=tmp_path) != get_idea_path(
        "A" * 120 + "y", output_root=tmp_path
    )
    assert get_paper_dir(
        "Alpha+Beta", timestamp="20260831_010203", output_root=tmp_path
    ) != get_paper_dir("Alpha Beta", timestamp="20260831_010203", output_root=tmp_path)


def test_safe_legacy_batch_spelling_and_verified_paper_resume_are_preserved(
    tmp_path: Path,
) -> None:
    assert get_batch_dir("MyBatch", output_root=tmp_path).name == "batch_MyBatch"

    legacy = tmp_path / "papers" / "paper_20260831_010203_my_idea_normal"
    legacy.mkdir(parents=True)
    (legacy / "idea.json").write_text('{"Name": "My Idea"}\n', encoding="utf-8")
    assert (
        get_paper_dir(
            "My Idea",
            paper_type="normal",
            timestamp="20260831_010203",
            output_root=tmp_path,
        )
        == legacy
    )


@pytest.mark.parametrize(
    "project_name", ["../escaped", "../../escaped", "/tmp/escaped"]
)
def test_relative_project_names_cannot_escape_output_root(
    tmp_path: Path, project_name: str
) -> None:
    with pytest.raises(ValueError, match="escapes"):
        get_project_dir(project_name, output_root=tmp_path)


@pytest.mark.parametrize(
    "project_name", ["CON", "nul.txt", "LPT9", "nested/CON", r"nested\NUL.txt"]
)
def test_windows_reserved_project_names_are_rejected_portably(
    tmp_path: Path, project_name: str
) -> None:
    with pytest.raises(ValueError, match="reserved on Windows"):
        get_project_dir(project_name, output_root=tmp_path)


@pytest.mark.parametrize("project_name", ["demo.", "nested/demo ", "bad:name"])
def test_windows_alias_and_invalid_project_components_are_rejected(
    tmp_path: Path, project_name: str
) -> None:
    with pytest.raises(ValueError, match="portable directory component"):
        get_project_dir(project_name, output_root=tmp_path)


def test_batch_duplicate_ideas_use_their_index_in_the_workspace_key(
    tmp_path: Path,
) -> None:
    from ai_scientist.apps.batch import _create_paper_workspace

    idea = {"Name": "duplicate", "Title": "Same content"}
    first = _create_paper_workspace(
        idea=idea,
        idea_index=1,
        paper_type="normal",
        timestamp="20260831_010203_000001",
        output_root=tmp_path,
    )
    second = _create_paper_workspace(
        idea=idea,
        idea_index=2,
        paper_type="normal",
        timestamp="20260831_010203_000001",
        output_root=tmp_path,
    )

    assert first["paper_dir"] != second["paper_dir"]
    assert "0001-duplicate-" in first["paper_dir"].name
    assert "0002-duplicate-" in second["paper_dir"].name


@pytest.mark.parametrize("attempt_id", [-1, True, "1", 1.5])
def test_invalid_attempt_ids_fail_closed(tmp_path: Path, attempt_id: object) -> None:
    with pytest.raises(ValueError, match="attempt_id"):
        get_experiment_dir("safe", attempt_id, output_root=tmp_path)  # type: ignore[arg-type]


@pytest.mark.parametrize("idea_index", [-1, True, "1", 1.5])
def test_invalid_idea_indexes_fail_closed(idea_index: object) -> None:
    with pytest.raises(ValueError, match="idea_index"):
        idea_storage_key({"Name": "safe"}, idea_index=idea_index)  # type: ignore[arg-type]


def test_content_identity_rejects_non_json_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="stable identity"):
        content_identity({"value": float("nan")})
    with pytest.raises(ValueError, match="stable identity"):
        content_identity({"value": object()})
