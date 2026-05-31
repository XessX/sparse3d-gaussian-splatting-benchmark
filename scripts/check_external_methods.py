import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir


def check_import(name: str, setup_paths: list[str], module: str, symbol: str | None = None) -> dict:
    original_path = list(sys.path)
    try:
        for rel_path in reversed(setup_paths):
            sys.path.insert(0, str(PROJECT_ROOT / rel_path))
        mod = importlib.import_module(module)
        if symbol:
            getattr(mod, symbol)
        return {"name": name, "module": module, "status": "ok", "error": None}
    except Exception as exc:
        return {"name": name, "module": module, "status": "failed", "error": repr(exc)}
    finally:
        sys.path = original_path


def main():
    checks = [
        check_import("vggt", [], "vggt.models.vggt", "VGGT"),
        check_import(
            "dust3r",
            ["external_methods/dust3r"],
            "dust3r.model",
            "AsymmetricCroCo3DStereo",
        ),
        check_import(
            "mast3r",
            ["external_methods/mast3r"],
            "mast3r.model",
            "AsymmetricMASt3R",
        ),
        check_import(
            "instantsplat",
            ["external_methods/InstantSplat"],
            "gaussian_renderer",
            "render",
        ),
        check_import(
            "gaussian_splatting_rasterizer",
            [],
            "diff_gaussian_rasterization",
            None,
        ),
        check_import("simple_knn", [], "simple_knn._C", None),
        check_import("fused_ssim", [], "fused_ssim", None),
    ]

    output = PROJECT_ROOT / "results" / "logs" / "external_method_imports.json"
    ensure_dir(output.parent)
    output.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(f"Saved external method import report to {output}")
    for check in checks:
        print(f"{check['name']}: {check['status']}")
        if check["error"]:
            print(f"  {check['error']}")


if __name__ == "__main__":
    main()
