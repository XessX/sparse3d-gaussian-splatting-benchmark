# Supplementary Information

This Supplementary Information supports the manuscript **"Benchmarking Gaussian Splatting Under Sparse and Degraded Views"**. The main manuscript retains six figures and two tables. Additional figures, tables, and protocol notes are listed here to keep the main paper within the Scientific Reports display-item limit.

## Supplementary Methods

The benchmark uses the same train/test separation rule for all evaluated public scenes: held-out test views are selected before sparse training splits are generated, and those held-out images are excluded from every training split. Degradation experiments alter selected training images only; held-out test images remain clean. This design separates reconstruction robustness from evaluation-image corruption.

External methods are evaluated according to the evidence they directly produce. COLMAP + 3DGS and InstantSplat are evaluated with held-out rendering metrics. VGGT, DUSt3R, and MASt3R are evaluated with aligned camera-center error and are therefore reported as pose proxy baselines rather than full rendering baselines.

## Supplementary Dataset Details

The evaluated public scenes are Tanks and Temples `tandt/train`, Tanks and Temples `tandt/truck`, Deep Blending `db/drjohnson`, and Deep Blending `db/playroom`. Raw public dataset images are not redistributed in the lightweight submission package. Users should download public datasets from official sources and verify license terms before redistributing any raw images.

Supplementary Note S1 provides a smartphone-scene protocol for future external validation. It defines scene naming, capture angles, image counts, train/test split rules, degradation plans, metadata CSV fields, folder structure, and benchmark commands.

## Supplementary View-Selection Details

The evaluated selection strategies are `uniform`, `random`, `quality`, `quality_diverse`, and `quality_image_diverse`. The `quality` selector ranks images using sharpness, exposure, entropy, and feature count. The pose-aware `quality_diverse` selector additionally uses COLMAP camera-center diversity and is treated as a diagnostic baseline. The pose-free `quality_image_diverse` selector combines image quality with color-histogram and ORB descriptor matching distances, making it more practical before camera poses are available.

## Supplementary Degradation Details

The 5-view degradation block evaluates JPEG Q20, JPEG Q40, motion blur kernel 15, 50% low-resolution simulation, underexposure 0.6, underexposure 0.4, and overexposure 1.4. Results are reported both as raw PSNR/SSIM/LPIPS and as changes relative to matched clean 5-view baselines.

## Supplementary Statistical Testing

Paired tests use matched scene, view-count, selection, and held-out image records. The main tests compare `quality_image_diverse` against `quality`, `quality_diverse`, `random`, and `uniform`. PSNR and SSIM are interpreted with higher values as better; LPIPS is interpreted with lower values as better. Bootstrap confidence intervals and Wilcoxon signed-rank p-values are reported in the main statistical table.

## Supplementary Figures

**Supplementary Fig. S1. Degradation LPIPS summary.** Caption: Mean held-out LPIPS for the 5-view degradation block. Source: `manuscript/figures/3dgs_degradation_5views_lpips_mean.png`. Supports: perceptual-quality analysis under degraded training images.

**Supplementary Fig. S2. Degradation PSNR summary.** Caption: Mean held-out PSNR for each degradation condition and selector. Source: `manuscript/figures/3dgs_degradation_5views_psnr_mean.png`. Supports: raw degradation metric interpretation alongside the main PSNR-drop figure.

**Supplementary Fig. S3. Degradation SSIM summary.** Caption: Mean held-out SSIM for each degradation condition and selector. Source: `manuscript/figures/3dgs_degradation_5views_ssim_mean.png`. Supports: structural-similarity interpretation of degradation robustness.

**Supplementary Fig. S4. 3DGS 5-view qualitative grid.** Caption: Example held-out render comparisons across 5-view selectors. Source: `manuscript/figures/3dgs_eval_5views_grid.png`. Supports: qualitative inspection of 3DGS sparse-view outputs.

**Supplementary Fig. S5. Clean 3DGS qualitative grid.** Caption: Clean 5-view render examples from the primary 3DGS benchmark. Source: `manuscript/figures/3dgs_eval_5views_grid_clean.png`. Supports: visual context for clean held-out metrics.

**Supplementary Fig. S6. Playroom qualitative grid.** Caption: Scene-specific 5-view render comparison for `db/playroom`. Source: `manuscript/figures/3dgs_eval_5views_grid_playroom.png`. Supports: scene-level qualitative analysis.

**Supplementary Fig. S7. Truck qualitative grid.** Caption: Scene-specific 5-view render comparison for `tandt/truck`. Source: `manuscript/figures/3dgs_eval_5views_grid_truck.png`. Supports: scene-level qualitative analysis.

**Supplementary Fig. S8. 5-view LPIPS by selector.** Caption: LPIPS comparison for 5-view 3DGS selection strategies. Source: `manuscript/figures/3dgs_eval_5views_lpips.png`. Supports: perceptual metric comparison of selectors.

**Supplementary Fig. S9. 5-view PSNR by selector.** Caption: PSNR comparison for 5-view 3DGS selection strategies. Source: `manuscript/figures/3dgs_eval_5views_psnr.png`. Supports: selector-performance comparison.

**Supplementary Fig. S10. 5-view SSIM by selector.** Caption: SSIM comparison for 5-view 3DGS selection strategies. Source: `manuscript/figures/3dgs_eval_5views_ssim.png`. Supports: structural-similarity comparison of selectors.

**Supplementary Fig. S11. LPIPS across view counts.** Caption: Held-out LPIPS across sparse-view counts. Source: `manuscript/figures/3dgs_eval_lpips_vs_views.png`. Supports: perceptual quality trends with increasing input views.

**Supplementary Fig. S12. PSNR across view counts.** Caption: Held-out PSNR across sparse-view counts for selected single-scene runs. Source: `manuscript/figures/3dgs_eval_psnr_vs_views.png`. Supports: view-count sensitivity analysis.

**Supplementary Fig. S13. SSIM across view counts.** Caption: Held-out SSIM across sparse-view counts. Source: `manuscript/figures/3dgs_eval_ssim_vs_views.png`. Supports: structural quality trends with increasing input views.

**Supplementary Fig. S14. Multiscene LPIPS across view counts.** Caption: Multiscene held-out LPIPS across view counts and selectors. Source: `manuscript/figures/3dgs_multiscene_lpips_vs_views.png`. Supports: multiscene perceptual metric trends.

**Supplementary Fig. S15. Multiscene SSIM across view counts.** Caption: Multiscene held-out SSIM across view counts and selectors. Source: `manuscript/figures/3dgs_multiscene_ssim_vs_views.png`. Supports: multiscene structural metric trends.

**Supplementary Fig. S16. Initial sparse points.** Caption: Initial point counts for sparse 3DGS splits. Source: `manuscript/figures/3dgs_sparse_initial_points.png`. Supports: interpretation of sparse reconstruction initialization.

**Supplementary Fig. S17. 3DGS runtime.** Caption: Runtime comparison for sparse-view 3DGS runs. Source: `manuscript/figures/3dgs_sparse_runtime.png`. Supports: practical feasibility analysis.

**Supplementary Fig. S18. Training PSNR.** Caption: Training PSNR summaries for sparse 3DGS runs. Source: `manuscript/figures/3dgs_sparse_train_psnr.png`. Supports: training-side monitoring.

**Supplementary Fig. S19. Foundation pose RMSE.** Caption: Camera-center RMSE for foundation-model pose proxy benchmarks. Source: `manuscript/figures/foundation_pose_5views_rmse.png`. Supports: proxy geometry comparison.

**Supplementary Fig. S20. Foundation pose runtime.** Caption: Runtime for foundation-model pose proxy benchmarks. Source: `manuscript/figures/foundation_pose_5views_runtime.png`. Supports: practical method comparison.

**Supplementary Fig. S21. InstantSplat qualitative grid.** Caption: Held-out rendering examples from the limited InstantSplat 5-view study. Source: `manuscript/figures/instantsplat_eval_5views_grid.png`. Supports: qualitative InstantSplat interpretation.

**Supplementary Fig. S22. InstantSplat LPIPS.** Caption: LPIPS results for the InstantSplat 5-view rendering study. Source: `manuscript/figures/instantsplat_eval_5views_lpips.png`. Supports: perceptual InstantSplat evaluation.

**Supplementary Fig. S23. InstantSplat PSNR.** Caption: PSNR results for the InstantSplat 5-view rendering study. Source: `manuscript/figures/instantsplat_eval_5views_psnr.png`. Supports: quantitative InstantSplat evaluation.

**Supplementary Fig. S24. InstantSplat runtime.** Caption: Runtime summary for InstantSplat evaluation. Source: `manuscript/figures/instantsplat_eval_5views_runtime.png`. Supports: feasibility analysis of InstantSplat.

**Supplementary Fig. S25. InstantSplat SSIM.** Caption: SSIM results for the InstantSplat 5-view rendering study. Source: `manuscript/figures/instantsplat_eval_5views_ssim.png`. Supports: structural InstantSplat evaluation.

**Supplementary Fig. S26. VGGT sparse pose RMSE.** Caption: VGGT camera-center RMSE across sparse-view settings. Source: `manuscript/figures/vggt_pose_sparse_rmse_vs_views.png`. Supports: view-count sensitivity of VGGT pose proxy results.

## Supplementary Tables

**Supplementary Table S1. Per-scene degradation robustness.** Source: `manuscript/tables/table_3dgs_degradation_eval.md`. Supports: scene-level degradation analysis.

**Supplementary Table S2. Mean degradation robustness.** Source: `manuscript/tables/table_3dgs_degradation_eval_mean.md`. Supports: average degradation metrics.

**Supplementary Table S3. Degradation change from clean.** Source: `manuscript/tables/table_3dgs_degradation_psnr_drop.md`. Supports: PSNR-drop interpretation in Fig. 5.

**Supplementary Table S4. Held-out 3DGS evaluation.** Source: `manuscript/tables/table_3dgs_heldout_eval.md`. Supports: per-split held-out metrics.

**Supplementary Table S5. Per-scene multiscene evaluation.** Source: `manuscript/tables/table_3dgs_multiscene_eval.md`. Supports: full clean benchmark row audit.

**Supplementary Table S6. Multiscene mean evaluation.** Source: `manuscript/tables/table_3dgs_multiscene_eval_mean.md`. Supports: clean performance summaries.

**Supplementary Table S7. 3DGS train runtime.** Source: `manuscript/tables/table_3dgs_train_runtime.md`. Supports: runtime and training metadata.

**Supplementary Table S8. Convergence check.** Source: `manuscript/tables/table_convergence_check.md`. Supports: Fig. 6 and iteration-count justification.

**Supplementary Table S9. Degradation expansion status.** Source: `manuscript/tables/table_degradation_expansion_todo.md`. Supports: audit that added degradation severities were completed.

**Supplementary Table S10. Foundation pose mean results.** Source: `manuscript/tables/table_foundation_pose_5views_mean.md`. Supports: foundation-model proxy benchmark.

**Supplementary Table S11. Foundation pose overall results.** Source: `manuscript/tables/table_foundation_pose_5views_overall.md`. Supports: aggregate foundation-model proxy comparison.

**Supplementary Table S12. InstantSplat per-scene evaluation.** Source: `manuscript/tables/table_instantsplat_eval_5views.md`. Supports: limited InstantSplat rendering study.

**Supplementary Table S13. InstantSplat mean evaluation.** Source: `manuscript/tables/table_instantsplat_eval_5views_mean.md`. Supports: InstantSplat summary metrics.

**Supplementary Table S14. InstantSplat smoke test.** Source: `manuscript/tables/table_instantsplat_smoke.md`. Supports: executable method setup validation.

**Supplementary Table S15. Main clean benchmark table.** Source: `manuscript/tables/table_main_clean_benchmark.md`. Supports: compact clean benchmark summary.

**Supplementary Table S16. Random-seed repeats.** Source: `manuscript/tables/table_random_seed_repeats.md`. Supports: random baseline variability in Fig. 3.

**Supplementary Table S17. VGGT sparse pose benchmark.** Source: `manuscript/tables/table_vggt_pose_sparse_mean.md`. Supports: sparse-view pose proxy trends.

## Reproducibility Notes

The lightweight submission package includes scripts, configuration files, generated CSV summaries, figures, tables, and audit reports. It excludes raw downloaded datasets, trained model checkpoints, external method repositories, local virtual environments, and bytecode/cache artifacts. Dataset license terms should be verified before public release of any raw dataset images or dataset-derived assets.

