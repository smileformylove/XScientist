import copy
import json
import os
from pathlib import Path
from typing import Type, TypeVar
import re
import uuid

import dataclasses_json


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def dumps_json(obj: dataclasses_json.DataClassJsonMixin):
    """Serialize dataclasses (such as Journals) to JSON."""
    from ..journal import Journal, Node

    if isinstance(obj, Journal):
        obj = copy.deepcopy(obj)
        node2parent = {}
        for n in obj.nodes:
            if n.parent is not None:
                # Handle both Node objects and string IDs
                parent_id = n.parent.id if isinstance(n.parent, Node) else n.parent
                node2parent[n.id] = parent_id
        for n in obj.nodes:
            n.parent = None
            n.children = set()

    obj_dict = obj.to_dict()

    if isinstance(obj, Journal):
        obj_dict["node2parent"] = node2parent
        obj_dict["__version"] = "2"

    return json.dumps(obj_dict, separators=(",", ":"))


def atomic_write_text(
    path: str | Path, content: str, *, encoding: str = "utf-8"
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temp_path.open("w", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
        _fsync_directory(path.parent)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_json(
    path: str | Path,
    payload,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    default=None,
) -> None:
    atomic_write_text(
        path,
        json.dumps(
            payload,
            indent=indent,
            ensure_ascii=ensure_ascii,
            default=default,
        ),
    )


def durable_append_text(
    path: str | Path,
    content: str,
    *,
    encoding: str = "utf-8",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode(encoding)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
    original_size = os.fstat(descriptor).st_size
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Failed to append experiment ledger row")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        try:
            os.ftruncate(descriptor, original_size)
            os.fsync(descriptor)
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)


def dump_json(obj: dataclasses_json.DataClassJsonMixin, path: Path):
    atomic_write_text(path, dumps_json(obj))


G = TypeVar("G", bound=dataclasses_json.DataClassJsonMixin)


def loads_json(s: str, cls: Type[G]) -> G:
    """Deserialize JSON to AIDE dataclasses."""
    from ..journal import Journal

    obj_dict = json.loads(s)
    obj = cls.from_dict(obj_dict)

    if isinstance(obj, Journal):
        id2nodes = {n.id: n for n in obj.nodes}
        for child_id, parent_id in obj_dict["node2parent"].items():
            id2nodes[child_id].parent = id2nodes[parent_id]
            id2nodes[child_id].__post_init__()
    return obj


def load_json(path: Path, cls: Type[G]) -> G:
    with open(path, "r") as f:
        return loads_json(f.read(), cls)


def parse_markdown_to_dict(content: str):
    """
    Reads a file that contains lines of the form:

        "Key": "Value",
        "Another Key": "Another Value",
        ...

    including possible multi-line values, and returns a Python dictionary.
    """

    pattern = r'"([^"]+)"\s*:\s*"([^"]*?)"(?:,\s*|\s*$)'

    matches = re.findall(pattern, content, flags=re.DOTALL)

    data_dict = {}
    for key, value in matches:
        data_dict[key] = value

    return data_dict
