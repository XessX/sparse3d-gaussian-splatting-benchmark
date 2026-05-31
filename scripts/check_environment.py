import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir
from src.utils.executables import find_executable


PYTHON_PACKAGES = [
    "cv2",
    "imageio",
    "lpips",
    "matplotlib",
    "numpy",
    "pandas",
    "PIL",
    "skimage",
    "torch",
    "torchvision",
    "yaml",
]

EXECUTABLES = ["git", "colmap", "nvidia-smi", "nvcc"]


def check_package(name: str) -> dict:
    spec = importlib.util.find_spec(name)
    return {"name": name, "available": spec is not None}


def run_version_command(command: list[str]) -> str | None:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except Exception:
        return None
    text = (proc.stdout or proc.stderr).strip()
    return text.splitlines()[0] if text else None


def check_executable(name: str) -> dict:
    path = find_executable(name)
    result = {"name": name, "available": path is not None, "path": path, "version": None}
    if path:
        if name == "nvidia-smi":
            result["version"] = run_version_command([path, "--query-gpu=name,driver_version", "--format=csv,noheader"])
        elif name == "nvcc":
            result["version"] = run_version_command([path, "--version"])
        else:
            result["version"] = run_version_command([path, "--version"])
    return result


def torch_status() -> dict:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    status = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            status["cuda_devices"].append(torch.cuda.get_device_name(idx))
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/logs/environment_check.json")
    args = parser.parse_args()

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": [check_package(pkg) for pkg in PYTHON_PACKAGES],
        "executables": [check_executable(exe) for exe in EXECUTABLES],
        "torch": torch_status(),
    }

    output = Path(args.output)
    ensure_dir(output.parent)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    missing_packages = [p["name"] for p in report["packages"] if not p["available"]]
    missing_executables = [e["name"] for e in report["executables"] if not e["available"]]

    print(f"Saved environment report to {output}")
    print(f"Missing Python packages: {missing_packages if missing_packages else 'none'}")
    print(f"Missing external executables: {missing_executables if missing_executables else 'none'}")
    print(f"Torch CUDA available: {report['torch'].get('cuda_available', False)}")


if __name__ == "__main__":
    main()
