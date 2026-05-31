import json
import sys
from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    env = load_json(PROJECT_ROOT / "results" / "logs" / "environment_check.json")
    imports = load_json(PROJECT_ROOT / "results" / "logs" / "external_method_imports.json")
    methods = load_json(PROJECT_ROOT / "results" / "logs" / "external_methods.json")
    colmap = load_json(
        PROJECT_ROOT / "results" / "method_outputs" / "demo_scene" / "colmap_sfm" / "run_report.json"
    )
    vggt = load_json(
        PROJECT_ROOT / "results" / "method_outputs" / "demo_scene" / "vggt_smoke" / "run_report.json"
    )
    real_vggt = load_json(
        PROJECT_ROOT / "results" / "method_outputs" / "tandt" / "train" / "vggt_smoke" / "run_report.json"
    )
    real_3dgs = load_json(
        PROJECT_ROOT / "results" / "method_outputs" / "tandt" / "train" / "3dgs_smoke" / "run_report.json"
    )
    degradation = load_json(
        PROJECT_ROOT / "datasets" / "processed" / "tandt" / "train" / "degraded" / "degradation_metadata.json"
    )
    manifest = load_json(PROJECT_ROOT / "results" / "logs" / "experiment_manifest.json")
    sparse_200 = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "3dgs_sparse_summary.csv")
    sparse_3000 = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "3dgs_sparse_summary_3000_clean.csv")
    if not sparse_3000:
        sparse_3000 = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "3dgs_sparse_summary_3000.csv")
    eval_3000 = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "3dgs_eval_summary_clean.csv")
    multiscene_eval = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "3dgs_multiscene_eval_3000.csv")
    multiscene_mean = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "3dgs_multiscene_eval_mean_3000.csv")
    degradation_eval = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "degradation" / "3dgs_degradation_eval_with_clean.csv")
    degradation_mean = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "degradation" / "3dgs_degradation_eval_mean.csv")
    foundation_pose = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "foundation" / "foundation_pose_5views_combined.csv")
    foundation_pose_mean = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "foundation" / "foundation_pose_5views_mean.csv")
    foundation_pose_overall = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "foundation" / "foundation_pose_5views_overall.csv")
    vggt_sparse_pose = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "foundation" / "vggt_pose_sparse_mean.csv")
    instantsplat_smoke = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "foundation" / "instantsplat_smoke.csv")
    instantsplat_eval = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "foundation" / "instantsplat_eval_5views.csv")
    instantsplat_eval_mean = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "foundation" / "instantsplat_eval_5views_mean.csv")
    random_repeats = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "random_seed_repeats.csv")
    convergence = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "convergence_check.csv")
    statistical_tests = load_csv_rows(PROJECT_ROOT / "results" / "csv" / "statistical_tests.csv")

    lines = [
        "# Sparse3D Setup Report",
        "",
        "## Environment",
    ]
    if env:
        lines.extend(
            [
                f"- Python: {env.get('python', '').split()[0]}",
                f"- Platform: {env.get('platform')}",
                f"- Torch: {env.get('torch', {}).get('version')}",
                f"- Torch CUDA available: {env.get('torch', {}).get('cuda_available')}",
                f"- CUDA devices: {', '.join(env.get('torch', {}).get('cuda_devices', [])) or 'none'}",
            ]
        )
        missing_exec = [e["name"] for e in env.get("executables", []) if not e.get("available")]
        missing_pkg = [p["name"] for p in env.get("packages", []) if not p.get("available")]
        lines.append(f"- Missing Python packages: {missing_pkg or 'none'}")
        lines.append(f"- Missing executables: {missing_exec or 'none'}")
        nvcc = next((e for e in env.get("executables", []) if e.get("name") == "nvcc"), None)
        if nvcc:
            lines.append(f"- NVCC path: {nvcc.get('path')}")

    lines.extend(["", "## External Source Repositories"])
    if methods:
        for method in methods:
            lines.append(
                f"- {method['method']}: {'present' if method['exists'] else 'missing'}"
                f" ({method.get('short_commit') or 'no commit'})"
            )

    lines.extend(["", "## Import Status"])
    if imports:
        for item in imports:
            line = f"- {item['name']}: {item['status']}"
            if item.get("error"):
                line += f" - {item['error']}"
            lines.append(line)

    lines.extend(["", "## Smoke-Test Outputs"])
    if colmap:
        lines.extend(
            [
                f"- COLMAP status: {colmap.get('status')}",
                f"- COLMAP models: {colmap.get('num_models')}",
                f"- COLMAP runtime_sec: {colmap.get('runtime_sec'):.3f}",
                f"- COLMAP output: {colmap.get('output_dir')}",
            ]
        )
    if vggt:
        lines.extend(
            [
                f"- VGGT device: {vggt.get('device')}",
                f"- VGGT inference_sec: {vggt.get('infer_sec'):.3f}",
                f"- VGGT peak_gpu_memory_mb: {vggt.get('peak_gpu_memory_mb'):.1f}",
                f"- VGGT output previews: {len(vggt.get('preview_files', []))}",
            ]
        )
    if real_vggt:
        lines.extend(
            [
                f"- Real-scene VGGT scene: {real_vggt.get('scene')}",
                f"- Real-scene VGGT inference_sec: {real_vggt.get('infer_sec'):.3f}",
                f"- Real-scene VGGT peak_gpu_memory_mb: {real_vggt.get('peak_gpu_memory_mb'):.1f}",
                f"- Real-scene VGGT previews: {len(real_vggt.get('preview_files', []))}",
            ]
        )
    if real_3dgs:
        lines.extend(
            [
                f"- Real-scene 3DGS smoke status: {real_3dgs.get('status')}",
                f"- Real-scene 3DGS iterations: {real_3dgs.get('iterations')}",
                f"- Real-scene 3DGS train PSNR@50: {real_3dgs.get('train_psnr_iter50'):.3f}",
                f"- Real-scene 3DGS point cloud: {real_3dgs.get('saved_point_cloud')}",
            ]
        )
    if sparse_200:
        lines.append(f"- Sparse 3DGS pilot rows at 200 iterations: {len(sparse_200)}")
    if sparse_3000:
        best = max(sparse_3000, key=lambda row: float(row.get("train_psnr") or 0))
        lines.append(f"- Sparse 3DGS train-only rows at 3000 iterations: {len(sparse_3000)}")
        lines.append(
            f"- Best 3000-iteration train PSNR: {best.get('split_name')} = {float(best.get('train_psnr')):.3f}"
        )
    if eval_3000:
        best_eval = max(eval_3000, key=lambda row: float(row.get("psnr") or 0))
        lines.append(f"- Sparse 3DGS held-out evaluation rows: {len(eval_3000)}")
        lines.append(
            f"- Best held-out PSNR: {best_eval.get('split')} = {float(best_eval.get('psnr')):.3f}"
        )
    if multiscene_eval:
        scenes = sorted({row.get("scene") for row in multiscene_eval if row.get("scene")})
        selections = sorted({row.get("selection") for row in multiscene_eval if row.get("selection")})
        best_multi = max(multiscene_eval, key=lambda row: float(row.get("psnr") or 0))
        lines.append(f"- Multi-scene 3DGS held-out evaluation rows: {len(multiscene_eval)}")
        lines.append(f"- Multi-scene scenes: {', '.join(scenes)}")
        lines.append(f"- Multi-scene selections: {', '.join(selections)}")
        lines.append(
            f"- Best multi-scene held-out PSNR: {best_multi.get('scene')} / {best_multi.get('split')} = {float(best_multi.get('psnr')):.3f}"
        )
    if multiscene_mean:
        best_mean = max(multiscene_mean, key=lambda row: float(row.get("psnr_mean") or 0))
        lines.append(
            f"- Best mean held-out PSNR: {best_mean.get('views')} views / {best_mean.get('selection')} = {float(best_mean.get('psnr_mean')):.3f}"
        )
    if degradation_eval:
        degraded_rows = [row for row in degradation_eval if row.get("condition") != "clean"]
        lines.append(f"- Degradation robustness rows including clean baselines: {len(degradation_eval)}")
        lines.append(f"- Degraded-input 3DGS runs: {len(degraded_rows)}")
    if degradation_mean:
        best_degraded = max(
            [row for row in degradation_mean if row.get("condition") != "clean"],
            key=lambda row: float(row.get("psnr_mean") or 0),
        )
        worst_degraded = min(
            [row for row in degradation_mean if row.get("condition") != "clean"],
            key=lambda row: float(row.get("psnr_mean") or 0),
        )
        lines.append(
            f"- Best degraded mean PSNR: {best_degraded.get('condition')} / {best_degraded.get('selection')} = {float(best_degraded.get('psnr_mean')):.3f}"
        )
        lines.append(
            f"- Worst degraded mean PSNR: {worst_degraded.get('condition')} / {worst_degraded.get('selection')} = {float(worst_degraded.get('psnr_mean')):.3f}"
        )
    if foundation_pose:
        methods = sorted({row.get("method") for row in foundation_pose if row.get("method")})
        failures = sum(row.get("status") != "success" for row in foundation_pose)
        lines.append(f"- Foundation 5-view pose rows: {len(foundation_pose)}")
        lines.append(f"- Foundation methods: {', '.join(methods)}")
        lines.append(f"- Foundation failed pose rows: {failures}")
    if foundation_pose_mean:
        best_foundation = min(
            foundation_pose_mean,
            key=lambda row: float(row.get("camera_rmse_mean") or 999),
        )
        lines.append(
            f"- Best 5-view foundation pose RMSE: {best_foundation.get('method')} / {best_foundation.get('selection')} = {float(best_foundation.get('camera_rmse_mean')):.3f}"
        )
    if foundation_pose_overall:
        best_overall_foundation = min(
            foundation_pose_overall,
            key=lambda row: float(row.get("camera_rmse_mean") or 999),
        )
        lines.append(
            f"- Best overall 5-view foundation pose RMSE: {best_overall_foundation.get('method')} = {float(best_overall_foundation.get('camera_rmse_mean')):.3f}"
        )
    if vggt_sparse_pose:
        high_view_rows = [row for row in vggt_sparse_pose if row.get("views") == "20"]
        best_vggt = min(
            high_view_rows or vggt_sparse_pose,
            key=lambda row: float(row.get("camera_rmse_mean") or 999),
        )
        lines.append(f"- VGGT sparse pose summary rows: {len(vggt_sparse_pose)}")
        lines.append(
            f"- Best VGGT 20-view pose RMSE: {best_vggt.get('selection')} = {float(best_vggt.get('camera_rmse_mean')):.3f}"
        )
    if instantsplat_smoke:
        latest = instantsplat_smoke[-1]
        lines.append(
            f"- InstantSplat smoke status: {latest.get('status')} "
            f"(init {float(latest.get('init_geo_sec') or 0):.3f}s, "
            f"train {float(latest.get('train_sec') or 0):.3f}s, "
            f"point cloud saved: {latest.get('saved_point_cloud_exists')})"
        )
    if instantsplat_eval:
        scenes = sorted({row.get("scene") for row in instantsplat_eval if row.get("scene")})
        failures = sum(row.get("status") != "success" for row in instantsplat_eval)
        best_eval = max(instantsplat_eval, key=lambda row: float(row.get("psnr") or 0))
        lines.append(f"- InstantSplat held-out rendering rows: {len(instantsplat_eval)}")
        lines.append(f"- InstantSplat scenes: {', '.join(scenes)}")
        lines.append(f"- InstantSplat failed rendering rows: {failures}")
        lines.append(
            f"- Best InstantSplat PSNR: {best_eval.get('scene')} / {best_eval.get('split')} = {float(best_eval.get('psnr')):.3f}"
        )
    if instantsplat_eval_mean:
        best_mean_instantsplat = max(instantsplat_eval_mean, key=lambda row: float(row.get("psnr_mean") or 0))
        lines.append(
            f"- Best mean InstantSplat PSNR: {best_mean_instantsplat.get('selection')} = {float(best_mean_instantsplat.get('psnr_mean')):.3f}"
        )
    if random_repeats:
        evaluated_random = sum(row.get("status") == "evaluated" for row in random_repeats)
        lines.append(f"- Repeated random-seed rows prepared: {len(random_repeats)}")
        lines.append(f"- Repeated random-seed rows evaluated: {evaluated_random}")
    if convergence:
        convergence_evaluated = [row for row in convergence if row.get("status") == "evaluated"]
        lines.append(f"- 3DGS convergence rows: {len(convergence)}")
        lines.append(f"- 3DGS convergence evaluated rows: {len(convergence_evaluated)}")
    if statistical_tests:
        all_view_tests = [row for row in statistical_tests if row.get("views") == "all"]
        lines.append(f"- Statistical test rows: {len(statistical_tests)}")
        lines.append(f"- All-view statistical comparison rows: {len(all_view_tests)}")

    lines.extend(["", "## Dataset Outputs"])
    train_images = PROJECT_ROOT / "datasets" / "raw" / "tandt" / "train" / "images"
    if train_images.exists():
        lines.append(f"- `tandt/train` images: {len(list(train_images.glob('*')))}")
    if degradation:
        lines.append(f"- Degraded `tandt/train` smoke images: {degradation.get('num_images')}")
    if manifest:
        missing = sum(row.get("status") == "missing_input" for row in manifest)
        lines.append(f"- Planned experiment rows: {len(manifest)}")
        lines.append(f"- Planned rows with missing inputs: {missing}")

    failed_imports = [item for item in (imports or []) if item.get("status") != "ok"]
    lines.extend(["", "## Remaining Blocker"])
    if failed_imports:
        for item in failed_imports:
            lines.append(f"- {item['name']}: {item.get('error')}")
    else:
        lines.append("- None for the starter pipeline: CUDA, COLMAP, VGGT, DUSt3R, MASt3R, InstantSplat imports, and GraphDeco native extensions are ready.")

    output = PROJECT_ROOT / "results" / "logs" / "setup_report.md"
    ensure_dir(output.parent)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved setup report to {output}")


if __name__ == "__main__":
    main()
