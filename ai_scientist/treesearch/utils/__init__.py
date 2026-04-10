import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger("ai-scientist")


def copytree(src: Path, dst: Path, use_symlinks=True):
    """
    Copy contents of `src` to `dst`. Unlike shutil.copytree, the dst dir can exist and will be merged.
    If src is a file, only that file will be copied. Optionally uses symlinks instead of copying.

    Args:
        src (Path): source directory
        dst (Path): destination directory
    """
    assert dst.is_dir()

    if src.is_file():
        dest_f = dst / src.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            (dest_f).symlink_to(src)
        else:
            shutil.copyfile(src, dest_f)
        return

    for f in src.iterdir():
        dest_f = dst / f.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            (dest_f).symlink_to(f)
        elif f.is_dir():
            shutil.copytree(f, dest_f)
        else:
            shutil.copyfile(f, dest_f)


def clean_up_dataset(path: Path):
    for item in path.rglob("__MACOSX"):
        if item.is_dir():
            shutil.rmtree(item)
    for item in path.rglob(".DS_Store"):
        if item.is_file():
            item.unlink()


def extract_archives(path: Path):
    """
    unzips all .zip files within `path` and cleans up task dir
    """
    processed: set[Path] = set()
    while True:
        zip_files = [
            zip_f
            for zip_f in path.rglob("*.zip")
            if zip_f.resolve() not in processed
        ]
        if not zip_files:
            break
        for zip_f in zip_files:
            processed.add(zip_f.resolve())
            _extract_single_archive(zip_f)


def _zip_matches_existing_file(zip_path: Path, existing_file: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = [info for info in zip_ref.infolist() if not info.is_dir()]
    except (OSError, zipfile.BadZipFile):
        return False
    if len(members) != 1:
        return False
    member = members[0]
    return (
        Path(member.filename).name == existing_file.name
        and existing_file.stat().st_size == member.file_size
    )


def _extract_single_archive(zip_f: Path) -> None:
    f_out_dir = zip_f.with_suffix("")

    if f_out_dir.exists():
        logger.debug(
            f"Skipping {zip_f} as an item with the same name already exists."
        )
        if f_out_dir.is_file() and _zip_matches_existing_file(zip_f, f_out_dir):
            zip_f.unlink()
        return

    logger.debug(f"Extracting: {zip_f}")
    f_out_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_f, "r") as zip_ref:
        zip_ref.extractall(f_out_dir)

    clean_up_dataset(f_out_dir)

    contents = list(f_out_dir.iterdir())

    if len(contents) == 1 and contents[0].name == f_out_dir.name:
        sub_item = contents[0]
        if sub_item.is_dir():
            logger.debug(f"Special handling (child is dir) enabled for: {zip_f}")
            for f in sub_item.rglob("*"):
                shutil.move(f, f_out_dir)
            sub_item.rmdir()
        elif sub_item.is_file():
            logger.debug(f"Special handling (child is file) enabled for: {zip_f}")
            sub_item_tmp = sub_item.rename(f_out_dir.with_suffix(".__tmp_rename"))
            f_out_dir.rmdir()
            sub_item_tmp.rename(f_out_dir)

    zip_f.unlink()


def preproc_data(path: Path):
    extract_archives(path)
    clean_up_dataset(path)
