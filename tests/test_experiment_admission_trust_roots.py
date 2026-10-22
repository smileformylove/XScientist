from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

import xscientist.research_git as research_git
from ai_scientist.utils.verifier_authority import load_ed25519_trust_store
from xscientist.research_git import ResearchGitError

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="Git is required")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _repository_with_linked_worktrees(tmp_path: Path) -> tuple[Path, Path, Path]:
    main = tmp_path.resolve() / "main"
    main.mkdir()
    _git(main, "init", "-q")
    _git(main, "config", "user.name", "Trust Boundary Test")
    _git(main, "config", "user.email", "trust-boundary@example.invalid")
    (main / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(main, "add", "seed.txt")
    _git(main, "commit", "-q", "-m", "seed")

    linked = tmp_path.resolve() / "linked"
    sibling = tmp_path.resolve() / "sibling"
    _git(main, "worktree", "add", "-q", "-b", "linked", str(linked))
    _git(main, "worktree", "add", "-q", "-b", "sibling", str(sibling))
    return main, linked, sibling


def _worktree_record(path: Path, *, branch: str = "test") -> str:
    return (
        f"worktree {path}\x00"
        f"HEAD {'0' * 40}\x00"
        f"branch refs/heads/{branch}\x00\x00"
    )


def _bounded_result(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        ["git", "worktree", "list", "--porcelain", "-z"],
        0,
        stdout,
        "",
    )


def test_linked_worktree_collects_every_external_trust_boundary(tmp_path: Path) -> None:
    main, linked, sibling = _repository_with_linked_worktrees(tmp_path)

    roots = research_git._repository_admission_forbidden_roots(linked)
    common_dir = (main / ".git").resolve()

    assert roots[0] == common_dir
    assert set(roots[1:]) == {main.resolve(), linked.resolve(), sibling.resolve()}
    assert len(roots) == 4

    for index, boundary in enumerate((main, linked, sibling, common_dir)):
        trust_store = boundary / f"authority-{index}" / "trust.json"
        trust_store.parent.mkdir()
        trust_store.write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="trust_store_inside_research_boundary"):
            load_ed25519_trust_store(trust_store, forbidden_roots=roots)

    sibling_trust = sibling / "authority-through-link" / "trust.json"
    sibling_trust.parent.mkdir()
    sibling_trust.write_text("{}\n", encoding="utf-8")
    linked_parent = tmp_path / "outside-looking-authority"
    linked_parent.symlink_to(sibling_trust.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="trust_store_inside_research_boundary"):
        load_ed25519_trust_store(
            linked_parent / "trust.json",
            forbidden_roots=roots,
        )


@pytest.mark.parametrize(
    "malformed",
    [
        "",
        "worktree /tmp/example\x00HEAD " + "0" * 40 + "\x00detached\x00",
        "\x00worktree /tmp/example\x00HEAD " + "0" * 40 + "\x00detached\x00\x00",
        "worktree /tmp/example\x00detached\x00\x00",
        "worktree /tmp/example\x00HEAD "
        + "0" * 40
        + "\x00branch refs/heads/main\x00detached\x00\x00",
        "worktree /tmp/example\x00HEAD "
        + "0" * 40
        + "\x00branch refs/heads/main\x00unexpected field\x00\x00",
        "worktree /tmp/example\nchild\x00HEAD " + "0" * 40 + "\x00detached\x00\x00",
        "worktree /tmp/example\x00injected\x00HEAD "
        + "0" * 40
        + "\x00detached\x00\x00",
        "worktree /tmp/\ufffd\x00HEAD " + "0" * 40 + "\x00detached\x00\x00",
    ],
)
def test_porcelain_z_parser_rejects_malformed_or_injected_nuls(
    malformed: str,
) -> None:
    with pytest.raises(ResearchGitError, match="Git worktree"):
        research_git._parse_git_worktree_list_porcelain_z(malformed)


def test_duplicate_worktree_path_fails_closed(tmp_path: Path) -> None:
    main, _, _ = _repository_with_linked_worktrees(tmp_path)
    duplicate_output = _worktree_record(main.resolve(), branch="main") * 2

    with mock.patch.object(
        research_git,
        "_run_git_bounded",
        return_value=_bounded_result(duplicate_output),
    ):
        with pytest.raises(ResearchGitError, match="duplicate path"):
            research_git._repository_admission_forbidden_roots(main)


@pytest.mark.parametrize(
    "unsafe_kind",
    ["missing", "file", "symlink", "symlink-parent"],
)
def test_unsafe_worktree_paths_fail_closed(tmp_path: Path, unsafe_kind: str) -> None:
    main, _, _ = _repository_with_linked_worktrees(tmp_path)
    unsafe_root = tmp_path / "unsafe"
    if unsafe_kind == "missing":
        candidate = unsafe_root
    elif unsafe_kind == "file":
        unsafe_root.write_text("not a directory\n", encoding="utf-8")
        candidate = unsafe_root
    elif unsafe_kind == "symlink":
        target = tmp_path / "real-worktree"
        target.mkdir()
        unsafe_root.symlink_to(target, target_is_directory=True)
        candidate = unsafe_root
    else:
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        (real_parent / "worktree").mkdir()
        alias_parent = tmp_path / "alias-parent"
        alias_parent.symlink_to(real_parent, target_is_directory=True)
        candidate = alias_parent / "worktree"

    output = _worktree_record(main.resolve(), branch="main") + _worktree_record(
        candidate,
        branch="unsafe",
    )
    with mock.patch.object(
        research_git,
        "_run_git_bounded",
        return_value=_bounded_result(output),
    ):
        with pytest.raises(ResearchGitError, match="Git worktree"):
            research_git._repository_admission_forbidden_roots(main)


def test_common_git_directory_with_symlink_parent_fails_closed(tmp_path: Path) -> None:
    main, _, _ = _repository_with_linked_worktrees(tmp_path)
    real_parent = tmp_path / "real-git-control"
    common_dir = real_parent / "common"
    common_dir.mkdir(parents=True)
    alias_parent = tmp_path / "git-control-alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    completed = subprocess.CompletedProcess(
        ["git", "rev-parse", "--git-common-dir"],
        0,
        f"{alias_parent / 'common'}\n",
        "",
    )

    with mock.patch.object(research_git, "_run_git", return_value=completed):
        with pytest.raises(ResearchGitError, match="symbolic link"):
            research_git._repository_common_git_dir(main)


@pytest.mark.parametrize("raw", ["", "\x00", ".git\x00\n", ".git\nextra\n"])
def test_common_git_directory_rejects_malformed_output(
    tmp_path: Path,
    raw: str,
) -> None:
    main, _, _ = _repository_with_linked_worktrees(tmp_path)
    completed = subprocess.CompletedProcess(
        ["git", "rev-parse", "--git-common-dir"],
        0,
        raw,
        "",
    )

    with mock.patch.object(research_git, "_run_git", return_value=completed):
        with pytest.raises(ResearchGitError, match="common directory path is invalid"):
            research_git._repository_common_git_dir(main)
