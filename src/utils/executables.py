from __future__ import annotations

import shutil
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]


LOCAL_EXECUTABLES = {
    "colmap": [
        PROJECT_ROOT / "external_methods" / "colmap-4.0.4-cuda" / "bin" / "colmap.exe",
        PROJECT_ROOT / "external_methods" / "colmap" / "bin" / "colmap.exe",
    ],
    "nvcc": [
        Path(os.environ.get("CUDA_PATH", "")) / "bin" / "nvcc.exe",
    ],
}


def find_executable(name: str) -> str | None:
    for path in LOCAL_EXECUTABLES.get(name, []):
        if path.exists():
            return str(path)
    resolved = shutil.which(name)
    return resolved
