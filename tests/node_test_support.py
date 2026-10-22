from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path


def find_node_executable() -> str | None:
    candidates = (
        shutil.which("node", path=os.defpath),
        "/opt/homebrew/bin/node",
        "/usr/local/bin/node",
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            resolved = Path(candidate).resolve(strict=True)
            metadata = resolved.stat()
        except OSError:
            continue
        if stat.S_ISREG(metadata.st_mode) and os.access(resolved, os.X_OK):
            return str(resolved)
    return None


NODE_EXECUTABLE = find_node_executable()
