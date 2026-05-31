# Benchmarking Gaussian Splatting Under Sparse and Degraded Views

## Title Page

**Title:** Benchmarking Gaussian Splatting Under Sparse and Degraded Views

**Authors:** Al Jubair Hossain  
**Affiliations:** American International University-Bangladesh (AIUB), 408/1 (Old KA 66/1), Kuratoli, Khilkhet, Dhaka 1229, Bangladesh  
**Corresponding author:** Al Jubair Hossain, [jubair.hossain@aiub.edu](mailto:jubair.hossain@aiub.edu)  
**ORCID:** 0009-0005-3498-5826

## Abstract

Sparse-view 3D reconstruction is important when only a few images are available, but the reliability of modern Gaussian Splatting and foundation-model pipelines under limited or degraded input remains unclear. We benchmarked COLMAP-initialized 3D Gaussian Splatting on four public scenes from Tanks and Temples and Deep Blending using 2, 3, 5, 8, 12, and 20 input views. The primary benchmark contains 120 clean evaluations across five view-selection strategies, plus 120 repeated random-seed evaluations. Quality-only selection was consistently weak. Pose-free image-diverse selection significantly improved over quality-only selection, but random, uniform, and pose-aware diversity remained stronger baselines. A completed 5-view degradation block with clean held-out test images showed that exposure shifts, especially severe underexposure, caused the largest performance losses. Foundation-model pose benchmarks for VGGT, DUSt3R, and MASt3R are reported as proxy metrics, and InstantSplat is evaluated in a limited 5-view rendering study. These results show that view diversity matters more than image quality alone, while practical pose-free selection remains challenging.

## Keywords

3D Gaussian Splatting; sparse-view reconstruction; novel-view synthesis; foundation models; image degradation; view selection; reproducible benchmark

## Introduction

Image-based 3D reconstruction and novel-view synthesis support cultural heritage documentation, robotics, digital twins, construction monitoring, extended reality, and scene-scale content creation. Many practical capture settings are sparse: a user may have only a few handheld images, some images may be compressed or blurred, and viewpoint coverage may be uneven. These low-resource capture conditions are common, but they are also where reconstruction pipelines are most likely to fail.

Sparse-view capture is not only a convenience problem. In field settings, users may be unable to walk around an object, revisit a site, use a calibrated rig, or capture dense image coverage under stable lighting. Heritage objects, shop fronts, monuments, and indoor corners can be partially occluded or accessible only from a restricted arc. Smartphone imagery can also introduce device-side processing, compression, rolling-shutter artifacts, exposure changes, and motion blur. A reconstruction method that performs well on dense benchmark captures can therefore behave differently when only a small number of views are available.

Neural radiance fields and related view-synthesis methods demonstrated that high-quality novel views can be recovered from posed images, but many such methods rely on dense or carefully sampled captures [14-17]. 3D Gaussian Splatting (3DGS) introduced an explicit Gaussian scene representation that can render high-quality views efficiently [1]. Standard 3DGS workflows usually depend on camera and sparse-point initialization from structure-from-motion tools such as COLMAP [2]. Sparse images can weaken feature matching, reduce geometric coverage, and produce incomplete or unstable reconstructions.

View selection is a key part of this problem. In sparse capture, choosing five images is not equivalent to choosing any five images: a set may contain sharp and well-exposed images but still cover only a narrow part of the scene. Conversely, a random or uniformly spaced set may include less ideal individual frames while covering a broader range of viewpoints. This means that quality-only heuristics should be tested against simple random and uniform baselines rather than assumed to be beneficial. It also means that diversity-aware selection should be separated into pose-aware and pose-free forms.

Recent foundation-model and pose-free reconstruction systems, including VGGT, DUSt3R, MASt3R, and InstantSplat, offer alternatives that are less dependent on conventional structure-from-motion [8-11]. These methods are promising, but their behavior under sparse and degraded input should be evaluated carefully. A benchmark is useful only if it distinguishes measured rendering evidence from proxy pose evidence and avoids claims that exceed the data.

This study asks: how reliable are Gaussian Splatting and foundation-model-guided reconstruction pipelines when input images are sparse or degraded, and how much does view selection matter?

The study is designed as a controlled benchmark rather than a new reconstruction architecture. Its purpose is to measure behavior under sparse and degraded inputs, audit whether simple selection heuristics help, and identify where claims should remain preliminary. This framing is important because foundation-model-guided reconstruction and Gaussian Splatting are developing quickly, and benchmark papers are useful only when they report both positive and negative findings with comparable baselines.

## Contributions

1. A reproducible four-scene sparse-view benchmark for COLMAP-initialized 3DGS.
2. A comparison of uniform, random, quality-only, pose-aware quality-diverse, and pose-free quality-image-diverse view selection.
3. A repeated random-seed analysis using five random seeds across all view counts.
4. A 5-view degradation robustness block with seven degraded training-image conditions and clean held-out test images.
5. Foundation-model camera-pose proxy benchmarks for VGGT, DUSt3R, and MASt3R.
6. A limited four-scene InstantSplat 5-view held-out rendering study.
7. Paired statistical tests, convergence checks, manuscript tables, figures, audit files, and reproducibility scripts generated from repository outputs.

## Results

### Benchmark Overview

The benchmark workflow is summarized in Fig. 1. Four public scenes were used: two Tanks and Temples scenes and two Deep Blending scenes [3,4]. For each scene, 20 views were held out for evaluation and excluded from all sparse training splits (Table 1). Each selected sparse split was trained with COLMAP-initialized 3DGS for 3000 iterations at resolution scale 4 and evaluated on clean held-out test images. Additional figures, qualitative grids, and per-method tables are listed in the Supplementary Information.

The four scenes provide a controlled starting point because they include existing COLMAP-compatible scene structure and represent both outdoor/scene-scale captures and indoor Deep Blending scenes. This makes them useful for comparing selection strategies without adding the confound of custom camera calibration or manual annotation. At the same time, four scenes are not sufficient for broad claims about all sparse-view reconstruction. The dataset summary therefore records source, image counts, split counts, and license status, and the discussion treats the results as controlled benchmark evidence that requires external validation.

### Sparse-View Gaussian Splatting

The primary clean benchmark contains 120 held-out 3DGS evaluations: 4 scenes x 6 view counts x 5 selection strategies. Mean held-out PSNR increased with view count for all selection strategies (Fig. 2). In the evaluated scenes, quality-only selection was consistently weak. Pose-aware quality-diverse selection improved strongly over quality-only selection at every view count, but it uses full-scene COLMAP camera centers and is therefore a diagnostic baseline rather than a deployable pre-reconstruction selector.

The pose-free quality-image-diverse selector uses image quality features and image-space diversity. It improved over quality-only selection but did not match random, uniform, or pose-aware quality-diverse baselines under this controlled benchmark. This supports a cautious conclusion: diversity matters, but practical pose-free diversity heuristics require broader validation.

This outcome is useful because it prevents an overly simple interpretation of quality-aware selection. Sharpness, exposure balance, entropy, and feature count are useful image descriptors, but they do not guarantee geometric complementarity. The pose-free image-diverse selector adds color-histogram and feature-matching distances so that selected images are less redundant, but image-space diversity is still an imperfect proxy for true viewpoint diversity. In the evaluated scenes, the method is best interpreted as a practical intermediate baseline: it avoids using ground-truth or full-scene poses, improves over quality-only selection, and still leaves room for stronger pose-free view-selection methods.

### Random-Seed Repeats

Repeated random splits for seeds 0, 1, 2, 3, and 4 were generated and evaluated for all four scenes and all six view counts, excluding held-out test images. This produced 120 additional random baseline evaluations. Mean PSNR increased from 10.569 dB at 2 views to 18.106 dB at 20 views, with standard deviations from 1.303 to 2.372 dB (Fig. 3). These results show that single random splits are informative but noisy, and that random selection remains a strong baseline in the evaluated scenes.

The repeated random-seed block also helps avoid a common benchmarking weakness. A single random split can be unusually favorable or unfavorable, especially at two or three views where one selected image can dominate the available coverage. Reporting mean and standard deviation across seeds gives a more stable picture of random-selection behavior and makes comparisons with deterministic selectors more transparent. This is why the manuscript treats random selection as a substantive baseline rather than as a weak control.

### Selector Comparison

At 5 views, quality-only selection had the weakest mean PSNR among the main selectors. Pose-free image-diverse selection improved over quality-only selection, while pose-aware quality-diverse, random, and uniform remained stronger baselines (Fig. 4). The paired statistical tests are summarized in Table 2. Across all matched scene, view-count, and held-out-image pairs, pose-free quality-image-diverse selection improved over quality-only selection by 1.134 dB PSNR on average, with a paired bootstrap 95% confidence interval of 0.866 to 1.406 dB and a Wilcoxon signed-rank p-value of 1.64e-14 [12,13].

However, quality-image-diverse selection was worse than pose-aware quality-diverse selection, random selection, and uniform selection in the same paired analysis. The mean PSNR differences were -1.634 dB against pose-aware quality-diverse, -2.251 dB against random, and -1.840 dB against uniform. For LPIPS, the same pattern held. These results preserve the main claim of the paper: quality-only sparse-view selection is weak, image diversity helps, and random/uniform baselines should remain central comparators.

### Degradation Robustness

The degradation block evaluated 5-view COLMAP + 3DGS training under degraded input images while keeping held-out test images clean. The completed degraded conditions were JPEG quality 20, JPEG quality 40, motion blur with kernel size 15, 50% low-resolution simulation, underexposure with intensity multipliers 0.6 and 0.4, and overexposure with multiplier 1.4. Reconstruction quality was measured using PSNR, SSIM, and LPIPS [5-7].

Degradation sensitivity is reported as PSNR drop relative to matched clean baselines (Fig. 5). Severe underexposure caused the largest average drop, followed by overexposure and moderate underexposure. JPEG compression, motion blur, and low-resolution simulation had smaller average effects in this 5-view setting, and small positive differences appear in a few cases because each degraded model is trained independently and sparse 3DGS has run-to-run variability.

Reporting degradation as a difference from the matched clean baseline is important because absolute PSNR varies across scenes and selectors. A raw PSNR table can obscure whether a condition is truly harmful or whether the underlying clean split was already weak. The drop-from-clean analysis instead asks how much quality is lost when the same sparse-view setting is trained with degraded inputs. Keeping the held-out test images clean further isolates training-input robustness: the measured loss reflects how degraded training evidence affects the reconstruction, not how a metric behaves on corrupted target images.

### Convergence Check

A convergence subset compared 3000 and 7000 training iterations for two scenes and four splits: `5views_uniform`, `5views_random`, `5views_quality_image_diverse`, and `20views_uniform`. The 7000-iteration runs did not improve held-out PSNR in this subset (Fig. 6). The mean PSNR change from 3000 to 7000 iterations was -2.531 dB, with all eight measured cases showing lower PSNR at 7000 iterations. This does not establish that 3000 iterations are optimal in every setting; rather, it supports using 3000 iterations as a consistent comparative setting for the current benchmark.

### Foundation-Model Pose Proxy Benchmark

VGGT, DUSt3R, and MASt3R were evaluated using a sparse-view camera-pose proxy metric [8-10]. Predicted camera centers were similarity-aligned to the full-scene COLMAP reference trajectory and evaluated by normalized camera-center RMSE. VGGT achieved the lowest mean aligned camera-center RMSE in this proxy benchmark. These results should not be interpreted as full novel-view synthesis results for VGGT, DUSt3R, or MASt3R unless downstream rendering is added.

The pose proxy is nevertheless useful because camera geometry is a prerequisite for many downstream reconstruction and rendering workflows. If a foundation model predicts camera centers that are more consistent with a reference trajectory, that can indicate better geometric organization under sparse input. However, pose consistency alone does not measure texture fidelity, occlusion handling, view-dependent effects, or the quality of rendered novel views. The proxy results are therefore reported alongside, not merged with, the 3DGS rendering metrics.

### InstantSplat Rendering Study

InstantSplat was evaluated in a limited 5-view rendering study across the same four scenes and four selection strategies [11]. Each run used MASt3R geometry initialization, 500 Gaussian training iterations, and 20 held-out test views. All 16 runs completed successfully. InstantSplat preserved coarse structure in some held-out views but produced visible smearing, ghosting, and incomplete geometry in several sparse-view cases. The evaluation uses a similarity-aligned test-camera bridge and should be treated as limited evidence rather than a definitive method ranking.

## Discussion

The results suggest three practical lessons for sparse-view 3DGS benchmarking. First, image quality alone is not enough: selecting only sharp, well-exposed, high-feature images can concentrate views in visually similar parts of the scene and reduce geometric coverage. Second, view diversity matters, but pose-aware diversity depends on camera estimates that may not be available before reconstruction. Third, random and uniform baselines remain strong and should not be treated as weak controls.

These lessons have direct implications for reproducible evaluation. A method that is evaluated only against a quality-only selector may appear stronger than it is, because quality-only selection can be a weak baseline in sparse settings. Conversely, a selector that improves over quality-only selection should not be described as generally superior unless it is also compared with repeated random seeds and uniform spacing. The benchmark therefore separates the practical question of whether a selector can be used before reconstruction from the diagnostic question of how much performance might be available if camera-center diversity were already known.

The degradation experiments show that exposure shifts deserve explicit reporting. Under this controlled benchmark, severe underexposure produced larger average losses than JPEG compression, motion blur, or the tested low-resolution simulation. Because held-out test images remained clean, these results isolate training-input degradation rather than measuring a mixture of training and evaluation corruption.

The exposure result is also relevant to smartphone and field capture. Exposure errors can reduce feature visibility, suppress texture, and change the appearance statistics that Gaussian optimization sees during training. Compression and blur remain important, but their effect may depend on scene content, camera motion, and the downsampling pipeline. The present degradation block should therefore be understood as a controlled first step: it identifies sensitivity patterns in the evaluated 5-view setting and provides scripts for extending the same protocol to additional views, scenes, and combined degradations.

The foundation-model results are useful but limited. VGGT, DUSt3R, and MASt3R pose results are proxy evidence, not rendered novel-view quality. InstantSplat is closer to a rendering comparison, but the current experiment is limited to 5-view inputs, 500 training iterations, and an aligned-test-camera bridge. Broader method-native rendering evaluation is needed before making stronger method-ranking claims.

The study also clarifies how foundation-model-guided methods should be incorporated into benchmark papers. Feed-forward or pose-free models may be fast and geometrically plausible, but their outputs must be evaluated with metrics that match the claim being made. Camera-center RMSE can support a statement about pose consistency; PSNR, SSIM, and LPIPS from held-out renders support a statement about novel-view synthesis; geometry metrics such as Chamfer distance or F-score would be needed for claims about reconstructed shape where ground truth is available. Keeping these evidence types separate reduces the risk of unfair comparisons.

## Methods

### Datasets

The benchmark uses four public scenes from two datasets: Tanks and Temples `train` and `truck`, and Deep Blending `drjohnson` and `playroom` [3,4]. Each scene includes images and a COLMAP sparse reconstruction. For each scene, 20 held-out test views were selected and excluded from all sparse training splits.

No additional Mip-NeRF 360, DTU, LLFF, or official 3DGS-compatible public scenes were available locally during the final polishing phase. The repository therefore includes a supplementary smartphone-scene protocol for external validation. New smartphone scenes should follow the same folder conventions, include metadata CSV files, reserve clean held-out test views before sparse-view selection, and then run the existing split, selection, degradation, rendering, and evaluation scripts without changing the train/test separation rule.

Raw public dataset images are not redistributed in the lightweight submission package. Users should obtain Tanks and Temples and Deep Blending data from official sources and then use the provided scripts to reproduce split generation, training, rendering, and analysis. Generated CSV summaries, split metadata, figures, and manuscript tables may be released with the code package, subject to final license verification for any dataset-derived assets.

### View Selection

Five selection strategies were evaluated. `uniform` selects evenly spaced images from the image sequence. `random` selects deterministic random samples from candidate training images. `quality` ranks images using sharpness, exposure, entropy, and ORB feature count. `quality_diverse` greedily combines image quality with COLMAP camera-center diversity and is therefore pose-aware. `quality_image_diverse` greedily combines image quality, color histogram distance, and ORB descriptor matching distance without using camera positions.

### Gaussian Splatting

Each selected split was materialized into a COLMAP-compatible scene folder containing selected training images, camera metadata, image metadata, and filtered sparse points. The primary 3DGS benchmark used 3000 training iterations with resolution scale 4 [1]. Evaluation scenes included the same training cameras plus 20 held-out cameras, with `test.txt` used to enforce test-view separation in rendering.

### Degradation Protocol

For degradation experiments, only training images were degraded. Held-out test images remained clean. Completed degradation conditions were JPEG Q20, JPEG Q40, motion blur kernel 15, low-resolution 50%, underexposure 0.6, underexposure 0.4, and overexposure 1.4. Degradation summaries report both raw PSNR/SSIM/LPIPS and changes relative to the matched clean 5-view baseline.

### Foundation-Model Pose Benchmark

VGGT, DUSt3R, and MASt3R were run on sparse image sets using official checkpoints [8-10]. Predicted camera centers were extracted, similarity-aligned to full-scene COLMAP camera centers, and evaluated with normalized RMSE. Runtime and peak GPU memory were recorded.

### InstantSplat Evaluation

InstantSplat was run on 5-view splits with MASt3R geometry initialization and 500 Gaussian training iterations [10,11]. Full-scene COLMAP test cameras were aligned into the InstantSplat coordinate frame using a similarity transform estimated from the five training camera centers. Rendered test views were compared with clean held-out images using PSNR, SSIM, and LPIPS.

### Statistical Analysis

Paired statistical comparisons were computed using matched scene, view-count, and held-out image pairs. Mean differences were reported with paired bootstrap confidence intervals and Wilcoxon signed-rank p-values [12,13]. For PSNR and SSIM, positive differences favor the target method. For LPIPS, negative differences favor the target method.

### Use of AI-assisted tools

AI-assisted tools were used for code organization, drafting support, and language polishing. All experimental design decisions, result interpretation, manuscript claims, and final content were reviewed and approved by the author(s). AI tools were not listed as authors.

## Limitations

First, pose-aware quality-diverse selection uses full-scene COLMAP camera centers, so it is not a deployable pre-reconstruction selector. It is included as a diagnostic baseline. The pose-free image-diverse selector is more practical but weaker than random and uniform baselines in the current results.

Second, VGGT, DUSt3R, and MASt3R are evaluated with pose proxy metrics, not full rendered novel-view metrics. Their results should not be directly compared to 3DGS PSNR/SSIM/LPIPS results unless downstream rendering is added.

Third, the InstantSplat benchmark is a limited 5-view study with 500 training iterations and an aligned-test-camera evaluation bridge. Longer training and method-native evaluation would be needed for a more complete InstantSplat comparison.

Fourth, the degradation block covers one 5-view setting. It includes multiple degradation severities, but it does not test every view count or every combined degradation.

Fifth, dataset coverage is limited to four scenes from two public datasets. Additional dataset families and smartphone scenes are planned as external validation, so the current findings should be interpreted as controlled benchmark evidence rather than broad performance conclusions.

Sixth, the current work emphasizes image-space metrics and pose proxies. Geometry metrics such as Chamfer distance and F-score would require datasets with verified ground-truth geometry and consistent evaluation masks. Adding such geometry evaluation on DTU or Tanks and Temples ground-truth subsets would strengthen future versions of the benchmark.

## Data Availability

The processed result tables, sparse-view split metadata, evaluation summaries, and manuscript figures generated for this study are available at https://doi.org/10.5281/zenodo.20476805. Raw Tanks and Temples and Deep Blending images are not redistributed with this release and should be obtained from the official dataset providers subject to their terms of use.

## Code Availability

The custom scripts used for sparse-view split generation, image-quality scoring, degradation synthesis, 3DGS evaluation summarization, statistical testing, and manuscript table generation are available at https://github.com/XessX/sparse3d-gaussian-splatting-benchmark and archived at https://doi.org/10.5281/zenodo.20476805. Third-party methods, including COLMAP, GraphDeco 3D Gaussian Splatting, VGGT, DUSt3R, MASt3R, and InstantSplat, should be obtained from their official repositories and used under their respective licenses.

## Acknowledgements

The author(s) received no specific funding for this work.

## Author Contributions

Al Jubair Hossain: conceptualization, methodology, software, validation, formal analysis, investigation, data curation, writing - original draft, writing - review and editing, visualization, and project administration.

## Competing Interests

The authors declare no competing interests.

## Supplementary Information

Supplementary Information includes additional figures, tables, result summaries, and the smartphone-scene external-validation protocol. The supplementary smartphone protocol is cited as Supplementary Note S1. Extra result curves and qualitative grids are listed as Supplementary Figs. S1-S26, and additional CSV-derived tables are listed as Supplementary Tables S1-S17.

## Figure Legends

**Fig. 1. Benchmark workflow.** Public scenes are split into held-out test views and sparse training views before selection. Degradation experiments alter training images only, while held-out test views remain clean.

**Fig. 2. Clean sparse-view 3DGS performance.** Mean held-out PSNR across view counts and view-selection strategies for the four evaluated scenes.

**Fig. 3. Repeated random-seed variability.** Mean and standard deviation of held-out PSNR across five random seeds, four scenes, and six view counts.

**Fig. 4. Five-view selector comparison.** Mean held-out PSNR for quality-only, pose-free image-diverse, pose-aware quality-diverse, random, and uniform 5-view selection.

**Fig. 5. Degradation sensitivity.** PSNR change relative to clean 5-view baselines for JPEG compression, blur, low-resolution simulation, underexposure, and overexposure.

**Fig. 6. Convergence subset.** Held-out PSNR comparison for 3000 and 7000 3DGS training iterations on selected scenes and splits.

## Table Legends

**Table 1. Dataset summary.** Public scene sources, image counts, held-out test-view counts, split counts, and dataset license-status notes.

**Table 2. Main paired statistical tests.** Paired comparisons for pose-free quality-image-diverse selection against quality-only, pose-aware quality-diverse, random, and uniform baselines.

## References

1. Kerbl, B., Kopanas, G., Leimkuehler, T. & Drettakis, G. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM Transactions on Graphics* 42, 139 (2023). https://arxiv.org/abs/2308.04079
2. Schoenberger, J. L. & Frahm, J.-M. Structure-from-Motion Revisited. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* 4104-4113 (2016). https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html
3. Knapitsch, A., Park, J., Zhou, Q.-Y. & Koltun, V. Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction. *ACM Transactions on Graphics* 36, 78 (2017). https://www.tanksandtemples.org/
4. Hedman, P., Philip, J., Price, T., Frahm, J.-M., Drettakis, G. & Brostow, G. Deep Blending for Free-Viewpoint Image-Based Rendering. *ACM Transactions on Graphics* 37, 257 (2018). https://discovery.ucl.ac.uk/id/eprint/10117776/
5. Huynh-Thu, Q. & Ghanbari, M. Scope of validity of PSNR in image/video quality assessment. *Electronics Letters* 44, 800-801 (2008). https://doi.org/10.1049/el:20080522
6. Wang, Z., Bovik, A. C., Sheikh, H. R. & Simoncelli, E. P. Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing* 13, 600-612 (2004). https://doi.org/10.1109/TIP.2003.819861
7. Zhang, R., Isola, P., Efros, A. A., Shechtman, E. & Wang, O. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* 586-595 (2018). https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html
8. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C. & Novotny, D. VGGT: Visual Geometry Grounded Transformer. arXiv:2503.11651 (2025). https://arxiv.org/abs/2503.11651
9. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B. & Revaud, J. DUSt3R: Geometric 3D Vision Made Easy. arXiv:2312.14132 (2023). https://arxiv.org/abs/2312.14132
10. Leroy, V., Cabon, Y. & Revaud, J. Grounding Image Matching in 3D with MASt3R. arXiv:2406.09756 (2024). https://arxiv.org/abs/2406.09756
11. Fan, Z. et al. InstantSplat: Sparse-view SfM-free Gaussian Splatting in Seconds. arXiv:2403.20309 (2024). https://arxiv.org/abs/2403.20309
12. Wilcoxon, F. Individual Comparisons by Ranking Methods. *Biometrics Bulletin* 1, 80-83 (1945). https://doi.org/10.2307/3001968
13. Efron, B. & Tibshirani, R. J. *An Introduction to the Bootstrap*. Chapman & Hall/CRC (1993).
14. Mildenhall, B. et al. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In *European Conference on Computer Vision* 405-421 (2020). https://arxiv.org/abs/2003.08934
15. Mildenhall, B. et al. Local Light Field Fusion: Practical View Synthesis with Prescriptive Sampling Guidelines. *ACM Transactions on Graphics* 38, 1-14 (2019). https://bmild.github.io/llff/
16. Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P. & Hedman, P. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* 5470-5479 (2022). https://openaccess.thecvf.com/content/CVPR2022/html/Barron_Mip-NeRF_360_Unbounded_Anti-Aliased_Neural_Radiance_Fields_CVPR_2022_paper.html
17. Jain, A., Tancik, M. & Abbeel, P. Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* 5885-5894 (2021). https://arxiv.org/abs/2104.00677

