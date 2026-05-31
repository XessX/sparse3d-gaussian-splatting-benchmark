# Sparse3D Scientific Reports Benchmark

Working title:

**Benchmarking Gaussian Splatting Under Sparse and Degraded Views**

This repository is a research codebase for a Scientific Reports-style computational benchmark on sparse-view 3D reconstruction, foundation-model-guided reconstruction, and Gaussian Splatting under degraded image conditions.

## Research idea

The project evaluates how modern 3D reconstruction and novel-view synthesis pipelines perform when input images are:

- sparse,
- compressed,
- blurred,
- low-resolution,
- underexposed or overexposed.

It also includes a lightweight contribution: **quality-aware sparse-view selection**, which ranks images using sharpness, exposure quality, entropy, feature richness, and optional viewpoint diversity.

## Datasets

Use public datasets only:

- Tanks and Temples: evaluated scenes `tandt/train`, `tandt/truck`
- Deep Blending: evaluated scenes `db/drjohnson`, `db/playroom`
- DTU MVS, Mip-NeRF 360, and LLFF remain candidate expansion datasets

Do not upload copyrighted random web images into the benchmark. Use official datasets with clear terms and cite them in the manuscript.

Raw public dataset images are not redistributed in the lightweight submission package. Users should download Tanks and Temples and Deep Blending scenes from official sources or the authorized 3DGS dataset package, then run the repository scripts against those local copies. Generated CSV summaries, split metadata, manuscript figures, tables, scripts, and audit reports may be released, but dataset license and redistribution terms require final manual verification before any public repository or Zenodo release that includes raw images or dataset-derived assets.

## Planned methods

External methods will be installed separately and called through wrapper scripts:

- COLMAP + 3D Gaussian Splatting
- VGGT
- DUSt3R / MASt3R
- InstantSplat
- Optional sparse-view 3DGS methods

The repository includes wrappers for heavy external methods plus a completed four-scene COLMAP + 3DGS benchmark, degradation analysis, repeated random-seed baseline, foundation-model pose proxy benchmark, and InstantSplat rendering study.

## Folder structure

```text
sparse3d-benchmark/
|-- configs/
|-- datasets/
|   |-- raw/
|   `-- processed/
|-- external_methods/
|-- scripts/
|-- src/
|   |-- data/
|   |-- degradation/
|   |-- selection/
|   |-- metrics/
|   |-- visualization/
|   `-- utils/
|-- results/
|   |-- csv/
|   |-- figures/
|   |-- logs/
|   `-- qualitative/
`-- manuscript/
    |-- figures/
    |-- tables/
    `-- draft/
```

## Setup

Create environment:

```bash
conda env create -f environment.yml
conda activate sparse3d
```

Or with pip:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

For CUDA PyTorch on this laptop, the working install command used was:

```powershell
python -m pip install --force-reinstall --no-cache-dir torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

## First workflow

1. Create a tiny synthetic demo scene:

```bash
python scripts/create_demo_scene.py --scene demo_scene --images 24
```

Or put your own sample scene into:

```text
datasets/raw/demo_scene/images/
```

2. Create sparse splits:

```bash
python scripts/create_sparse_splits.py --scene demo_scene --views 2 3 5 8 12 20
```

3. Apply degradations:

```bash
python scripts/apply_degradations.py --scene demo_scene
```

4. Run quality-aware view selection:

```bash
python scripts/run_view_selection.py --scene demo_scene --views 5
```

5. Run placeholder baseline:

```bash
python scripts/run_baseline_placeholder.py --scene demo_scene --method placeholder_3dgs
```

6. Generate sample result figures:

```bash
python scripts/generate_figures.py
```

7. Check the local environment:

```bash
python scripts/check_environment.py
```

8. Create a reproducible experiment manifest:

```bash
python scripts/create_experiment_manifest.py --scene demo_scene
```

9. Run a local COLMAP sparse reconstruction smoke test:

```bash
python scripts/run_colmap_sfm.py --scene demo_scene
```

10. Record external method source commits:

```bash
python scripts/record_external_methods.py
```

11. Check which external methods import cleanly:

```bash
python scripts/check_external_methods.py
```

12. Run a small VGGT GPU smoke test:

```bash
python scripts/run_vggt_smoke.py --scene demo_scene --num-images 2
```

13. Generate a consolidated setup report:

```bash
python scripts/generate_setup_report.py
```

## Current real benchmark outputs

The current machine has CUDA PyTorch, COLMAP, GraphDeco 3DGS native extensions, VGGT, DUSt3R, MASt3R, and InstantSplat imports working.

The completed benchmark evaluates COLMAP + 3DGS at 3000 iterations on four public scenes:

- `tandt/train`
- `tandt/truck`
- `db/drjohnson`
- `db/playroom`

For each scene, it uses:

- view counts: 2, 3, 5, 8, 12, 20,
- view selection: uniform, random, quality-aware, pose-aware quality-diverse, pose-free quality-image-diverse,
- held-out evaluation: 20 test views excluded from all sparse training splits.

It also includes 120 repeated random-seed evaluations using seeds 0-4 for all four scenes and all six view counts.

The foundation-model benchmark is also integrated as a camera-pose reconstruction proxy:

- VGGT: 2, 3, 5, 8, 12, and 20 views over all four scenes and four selection strategies,
- DUSt3R: 5-view benchmark over all four scenes and four selection strategies,
- MASt3R: 5-view benchmark over all four scenes and four selection strategies.
- InstantSplat: executable 5-view held-out rendering study over all four scenes and four selection strategies, including MASt3R geometry initialization, 500-iteration Gaussian training, 20 held-out renders per run, and PSNR/SSIM/LPIPS evaluation.

These foundation-model results measure aligned camera-center error against the full COLMAP reference trajectory. They are not a replacement for full novel-view rendering metrics, but they provide a reproducible robustness signal for pose-free/foundation reconstruction under sparse inputs.

Important output files:

```text
results/csv/3dgs_sparse_summary_3000_clean.csv
results/csv/3dgs_eval_summary_clean.csv
results/csv/3dgs_multiscene_train_3000.csv
results/csv/3dgs_multiscene_eval_3000.csv
results/csv/3dgs_multiscene_eval_mean_3000.csv
results/csv/quality_image_diverse_splits.csv
results/csv/random_seed_repeats.csv
results/csv/random_seed_repeats_summary.csv
results/csv/convergence_check.csv
results/csv/statistical_tests.csv
results/csv/degradation/3dgs_degradation_eval_with_clean.csv
results/csv/degradation/3dgs_degradation_eval_mean.csv
results/csv/degradation/3dgs_degradation_psnr_drop.csv
results/csv/foundation/foundation_pose_5views_combined.csv
results/csv/foundation/foundation_pose_5views_mean.csv
results/csv/foundation/foundation_pose_5views_overall.csv
results/csv/foundation/vggt_pose_sparse_all.csv
results/csv/foundation/vggt_pose_sparse_mean.csv
results/csv/foundation/instantsplat_smoke.csv
results/csv/foundation/instantsplat_eval_5views.csv
results/csv/foundation/instantsplat_eval_5views_mean.csv
results/figures/3dgs_3000_clean/
results/figures/3dgs_eval_clean/
results/figures/3dgs_multiscene/
results/figures/3dgs_degradation/
results/figures/foundation/
results/qualitative/3dgs_eval_5views_grid_clean.png
results/qualitative/instantsplat_eval_5views_grid.png
manuscript/tables/table_3dgs_heldout_eval.md
manuscript/tables/table_3dgs_train_runtime.md
manuscript/tables/table_3dgs_multiscene_eval_mean.md
manuscript/tables/table_3dgs_degradation_eval_mean.md
manuscript/tables/table_dataset_summary.md
manuscript/tables/table_random_seed_repeats.md
manuscript/tables/table_convergence_check.md
manuscript/tables/table_statistical_tests.md
manuscript/tables/table_3dgs_degradation_psnr_drop.md
manuscript/tables/table_foundation_pose_5views_mean.md
manuscript/tables/table_foundation_pose_5views_overall.md
manuscript/tables/table_vggt_pose_sparse_mean.md
manuscript/tables/table_instantsplat_smoke.md
manuscript/tables/table_instantsplat_eval_5views_mean.md
results/logs/setup_report.md
```

Reproduce the clean COLMAP + 3DGS benchmark:

```powershell
python scripts\run_3dgs_sparse.py --scene "tandt/train" --splits 2views_uniform 2views_random 2views_quality 3views_uniform 3views_random 3views_quality 5views_uniform 5views_random 5views_quality 8views_uniform 8views_random 8views_quality 12views_uniform 12views_random 12views_quality 20views_uniform 20views_random 20views_quality --iterations 3000 --resolution 4 --csv results/csv/3dgs_sparse_runs_3000_clean.csv
python scripts\create_quality_diverse_splits.py --scene "tandt/train"
python scripts\run_3dgs_sparse.py --scene "tandt/train" --splits 2views_quality_diverse 3views_quality_diverse 5views_quality_diverse 8views_quality_diverse 12views_quality_diverse 20views_quality_diverse --iterations 3000 --resolution 4 --csv results/csv/3dgs_sparse_runs_3000_clean.csv
python scripts\render_eval_3dgs.py
python scripts\summarize_3dgs_sparse.py --input results\csv\3dgs_sparse_runs_3000_clean.csv --output results\csv\3dgs_sparse_summary_3000_clean.csv --fig-dir results\figures\3dgs_3000_clean
python scripts\summarize_eval_metrics.py --csv results\csv\3dgs_eval_summary_clean.csv --figures results\figures\3dgs_eval_clean
python scripts\generate_qualitative_grid.py --output results\qualitative\3dgs_eval_5views_grid_clean.png
python scripts\export_manuscript_tables.py
python scripts\generate_setup_report.py
```

After running additional scenes, aggregate all COLMAP + 3DGS results:

```powershell
python scripts\collect_multiscene_3dgs_results.py
python scripts\export_manuscript_tables.py
```

Run the 5-view degradation robustness block:

```powershell
python scripts\run_3dgs_degradation.py --iterations 3000 --resolution 4 --csv results/csv/3dgs_degradation_runs_3000.csv
python scripts\collect_degradation_results.py
python scripts\export_manuscript_tables.py
```

Run the completed added degradation severities:

```powershell
python scripts\run_degradation_expansion_eval.py --conditions jpeg_q40 under_0.4 over_1.4
python scripts\collect_degradation_results.py
python scripts\create_degradation_expansion_todo.py
python scripts\export_manuscript_tables.py
```

Run the foundation-model camera-pose benchmark summaries:

```powershell
python scripts\run_vggt_pose_benchmark.py --csv results/csv/foundation/vggt_pose_sparse_all.csv
python scripts\run_dust3r_pose_benchmark.py --niter 100 --csv results/csv/foundation/dust3r_pose_5views.csv
python scripts\run_mast3r_pose_benchmark.py --clear-cache --csv results/csv/foundation/mast3r_pose_5views.csv
python scripts\collect_foundation_pose_results.py
python scripts\run_instantsplat_smoke.py --scene "tandt/train" --split 5views_quality_diverse --views 5 --train-iterations 100 --resolution 4 --csv results/csv/foundation/instantsplat_smoke.csv
python scripts\run_instantsplat_eval_benchmark.py --train-iterations 500 --resolution 4 --optim-test-pose-iter 0 --csv results/csv/foundation/instantsplat_eval_5views.csv
python scripts\summarize_instantsplat_eval.py
python scripts\export_manuscript_tables.py
python scripts\generate_setup_report.py
```

Run the submission-hardening additions:

```powershell
python scripts\create_quality_image_diverse_splits.py
python scripts\run_3dgs_sparse.py --scene "tandt/train" --splits 2views_quality_image_diverse 3views_quality_image_diverse 5views_quality_image_diverse 8views_quality_image_diverse 12views_quality_image_diverse 20views_quality_image_diverse --iterations 3000 --resolution 4 --csv results/csv/3dgs_sparse_runs_3000_image_diverse.csv
python scripts\render_eval_3dgs.py --scene "tandt/train" --splits 2views_quality_image_diverse 3views_quality_image_diverse 5views_quality_image_diverse 8views_quality_image_diverse 12views_quality_image_diverse 20views_quality_image_diverse --iteration 3000 --resolution 4 --skip-existing
python scripts\collect_multiscene_3dgs_results.py
python scripts\create_random_seed_repeats.py
python scripts\run_random_seed_repeats_eval.py
python scripts\run_convergence_check.py --run-missing
python scripts\run_statistical_tests.py
python scripts\create_dataset_summary_table.py
python scripts\create_degradation_expansion_todo.py
python scripts\create_submission_package.py
```

## Manuscript plan

The manuscript draft outline is in:

```text
manuscript/draft/scientific_reports_outline.md
```

Dataset and method source notes are in:

```text
manuscript/draft/dataset_and_method_sources.md
```

The next project task is in:

```text
manuscript/draft/task_02_dataset_and_method_setup.md
```

## Notes

This repository now contains a clean four-scene COLMAP + 3DGS sparse-view benchmark with five primary view-selection strategies, 120 repeated random-seed evaluations, a four-scene 5-view degradation robustness block with seven degraded training-image conditions, foundation-model camera-pose benchmarks for VGGT, DUSt3R, and MASt3R, a four-scene InstantSplat 5-view held-out rendering study, statistical tests, and a convergence subset. The next strongest additions before submission are more datasets/scenes, downstream rendering for foundation-model baselines, path normalization for public release, and geometry metrics on datasets with official ground-truth geometry.
