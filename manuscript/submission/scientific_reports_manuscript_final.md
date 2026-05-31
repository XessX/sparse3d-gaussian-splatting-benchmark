Benchmarking Gaussian Splatting Under Sparse and Degraded Views

Title Page

Title: Benchmarking Gaussian Splatting Under Sparse and Degraded Views

Authors: Al Jubair Hossain

Affiliations: American International University-Bangladesh (AIUB), 408/1 (Old KA 66/1), Kuratoli, Khilkhet, Dhaka 1229, Bangladesh

Corresponding author: Al Jubair Hossain, jubair.hossain@aiub.edu

ORCID: 0009-0005-3498-5826

Abstract

The performance of modern Gaussian Splatting and foundation-model pipelines with limited or degraded input remains insufficiently evaluated, while sparse-view 3D reconstruction is crucial when only a few images are available. We tested COLMAP-initialized 3D Gaussian Splatting on four public scenes from Tanks and Temples and Deep Blending using 2, 3, 5, 8, 12, and 20 input views. There are 120 clean evaluations in the primary benchmark, and 120 evaluations with repeated random seeds. Quality-only selection was poor across the evaluated settings. Pose-free image-diverse selection was significantly better than quality-only selection; however, random, uniform, and pose-aware diversity remained strong baselines. A 5-view degradation block with clean held-out test images revealed exposure-shift losses, particularly under severe underexposure. Proxy metrics are reported for the foundation models VGGT, DUSt3R, and MASt3R, and InstantSplat is assessed in a small 5-view rendering study. These results demonstrate that view diversity is more important than image quality alone and that practical pose-free view selection remains challenging.

Keywords

3D Gaussian Splatting; sparse-view reconstruction; novel-view synthesis; foundation models; image degradation; view selection; reproducible benchmark

Introduction

Image based 3D reconstruction and novel-view synthesis enable documentation of cultural heritage, robotics, digital twin, construction monitoring, extended reality and content creation at the scale of a scene. In many real-world situations, there are only a few hand-held images, images are compressed or blurred, and there is uneven viewpoint coverage. The conditions in which these captures are made are low resource conditions, but they are also the ones in which reconstruction pipelines are most likely to fail.

Sparse-view capture isn't just a convenience issue. The user may not be able to step around an object, return to a site, utilize a calibrated rig, or shoot dense coverage in a stable lighting environment in field settings. Elements of the heritage, shop fronts, monuments and corners can be partially occluded or only partially available from a limited arc. Smartphone imaging may also add device side processing, compression and rolling shutter artifacts, exposure variations, and motion blur. If a reconstruction method achieves good performance on a dense benchmark capture, then it will behave differently if only a few views are used.

Many view-synthesis approaches require dense or carefully sampled captures [14-17] and neural radiance fields showed that high-quality novel views can be recovered from posed images. To achieve high quality views efficiently, 3D Gaussian Splatting (3DGS) provided an explicit Gaussian scene representation [1]. Typical 3DGS workflows rely on camera and sparse-point initialization from structure-from-motion (SfM) techniques like COLMAP [2]. Low density images may lead to poor feature matching, insufficient geometric coverage, and incomplete or unstable reconstruction.

One of the critical factors of this problem is view selection. In sparse capture, it's not the same to select five images as to select any five images: a set could include sharp, well-exposed images, but it might only capture a small section of the scene. On the other hand, a random or uniformly spaced set can be a less than ideal set of individual frames, but cover a wider range of viewpoints. Quality-only heuristics should be evaluated against simple random and uniform baselines, and should not be assumed to be helpful. It also implies the separation of pose-aware and pose-free forms of diversity-aware selection.

Recent foundation-model and pose-free reconstruction systems, including VGGT [8], DUSt3R [9], MASt3R [10], and InstantSplat [11], offer alternatives that are less dependent on conventional structure-from-motion. These methods are promising, but they need to be tested carefully for their behavior under sparse and degraded input. A benchmark is useful only when it separates measured rendering evidence from proxy pose evidence and avoids claims beyond what is measured.

The question in this study is how reliable Gaussian Splatting and foundation-model-guided reconstruction pipelines are when input images are sparse or degraded, and how much the choice of views affects accuracy.

The study is not presented as a new reconstruction architecture, but as a controlled benchmark. It is designed to assess behavior under sparse and degraded inputs, evaluate the efficacy of simple selection heuristics, and determine where claims should remain provisional. This framing is crucial because both foundation-model-guided reconstruction and Gaussian Splatting are rapidly advancing fields, and benchmark papers also need to report both positive and negative results with similar baselines.

Contributions

1. A reproducible four-scene sparse-view benchmark for COLMAP-initialized 3DGS.

2. A comparison of uniform, random, quality-only, pose-aware quality-diverse, and pose-free quality-image-diverse view selection.

3. A repeated random-seed analysis using five random seeds across all view counts.

4. A 5-view degradation robustness block with seven degraded training-image conditions and clean held-out test images.

5. Foundation-model camera-pose proxy benchmarks for VGGT, DUSt3R, and MASt3R.

6. A limited four-scene InstantSplat 5-view held-out rendering study.

7. Paired statistical tests, convergence checks, manuscript tables, figures, audit files, and reproducibility scripts generated from repository outputs.

Results

Benchmark Overview

The benchmark workflow is summarized in Fig. 1. Four public scenes were used: two Tanks and Temples scenes and two Deep Blending scenes [3,4]. For each scene, 20 views were held out for evaluation and excluded from all sparse training splits (Table 1). Each selected sparse split was trained with COLMAP-initialized 3DGS for 3000 iterations at resolution scale 4 and evaluated on clean held-out test images. Additional figures, qualitative grids, and per-method tables are listed in the Supplementary Information.

The four scenes provide a controlled starting point because they include existing COLMAP-compatible scene structure and represent both outdoor/scene-scale captures and indoor Deep Blending scenes. This makes them useful for comparing selection strategies without adding the confound of custom camera calibration or manual annotation. At the same time, four scenes are not sufficient for broad claims about all sparse-view reconstruction. The dataset summary therefore records source, image counts, split counts, and license status, and the discussion treats the results as controlled benchmark evidence that requires external validation.

Sparse-View Gaussian Splatting

The primary clean benchmark is composed of 120 held-out 3DGS evaluations, which are 4 scenes x 6 view counts x 5 selection strategies. For all selection strategies, the Mean held-out PSNR increased as the number of views increased (Fig. 2). Quality-only selection was always low in the scenes evaluated. Pose-aware quality-diverse selection improved very strongly over quality-only selection at all view counts, but it relies on full-scene COLMAP camera centers and is therefore not a viable pre-reconstruction selector.

The image quality features and image space diversity are used in the pose-free quality-image-diverse selector. It enhanced the quality-only selection but failed to equal the random, uniform and pose-aware quality-diverse baselines under this controlled benchmark. This leads to a tentative conclusion: diversity is important but there is a need for wider validation of practical, pose-free diversity heuristics.

The good thing about that is that it allows for a non-simplistic understanding of quality-aware selection. Among the image descriptors that are useful, but not necessarily equivalent to geometric complementarity, are sharpness, exposure balance, entropy and feature count. The image-diverse selector is posed-free, and calculates color-histogram and feature-matching distances, to reduce redundancy among the selected images, but image-space diversity is still not a perfect measure of the true-viewpoint diversity. The method is best understood in the evaluated scenes as a practical intermediate baseline, that is, it does not require the use of ground-truth or full-scene poses, outperforms quality-only selection, and still offers potential for better pose-free view-selection methods.

Random-Seed Repeats

Seeds 0, 1, 2, 3, and 4 were randomly split and evaluated, for each of the four scenes and the six view counts, without held-out test images. This yielded 120 extra random baseline assessments. Mean PSNR increased from 10.569 dB at 2 views to 18.106 dB at 20 views, with standard deviations from 1.303 to 2.372 dB (Fig. 3). The results demonstrate that while single random splits are informative, they are noisy, and that random selection is a good baseline in the scenes evaluated.

A repeated block with random seed also minimizes a common weakness in benchmarking. At two or three views, one image may be a very good or very bad choice when the view is split at random, and one image may be very good or very bad with the split at a single random point. Reporting the mean and standard deviation across seeds provides a more consistent view of the random selection behavior, and allows more easily comparison with deterministic selectors. That is why random selection is given the status of a substantive baseline in the manuscript, rather than being a feeble control.

Selector Comparison

At 5 views, quality-only selection had the weakest mean PSNR among the main selectors. Pose-free image-diverse selection improved over quality-only selection, while pose-aware quality-diverse, random, and uniform remained stronger baselines (Fig. 4). The paired statistical tests are summarized in Table 2. Across all matched scene, view-count, and held-out-image pairs, pose-free quality-image-diverse selection improved over quality-only selection by 1.134 dB PSNR on average, with a paired bootstrap 95% confidence interval of 0.866 to 1.406 dB and a Wilcoxon signed-rank p-value of 1.64e-14 [12,13].

However, quality-image-diverse selection was worse than pose-aware quality-diverse selection, random selection, and uniform selection in the same paired analysis. The mean PSNR differences were -1.634 dB against pose-aware quality-diverse, -2.251 dB against random, and -1.840 dB against uniform. For LPIPS, the same pattern held. These results preserve the main claim of the paper: quality-only sparse-view selection is weak, image diversity helps, and random/uniform baselines should remain central comparators.

Degradation Robustness

The degradation block was employed to test the performance of 5-view COLMAP + 3DGS training under degraded training images while keeping the held-out test images clean. The completed degradation conditions were JPEG quality 20, JPEG quality 40, motion blur with kernel size 15, 50% low-resolution simulation, underexposure with intensity multipliers 0.6 and 0.4, and overexposure with an intensity multiplier of 1.4. The quality of the reconstruction was assessed by PSNR, SSIM, and LPIPS [5-7].

The degradation sensitivity is expressed as the drop of PSNR from matched clean baselines (Fig. 5). The greatest average decrease was for severe underexposure, followed by overexposure and moderate underexposure. JPEG compression, motion blur and low resolution simulation had smaller average impacts in this 5-view setting, and the small positive differences in a few cases were as a result of each degraded model being trained separately and the inherent run-to-run variation in the sparse 3DGS.

It is important to report degradation as a difference from the matched clean baseline since absolute PSNR can differ from one scene to another and from one selector to another. Without this matched drop-from-clean analysis, it may be difficult to determine whether a condition is a real problem or whether the clean split was already weak. The drop-from-clean analysis asks what quality is lost when the same sparse-view setting is trained with degraded inputs. Keeping the held-out test images clean further isolates the robustness of training evidence: the measured loss is not a measure of degraded target images, but of degraded training evidence.

Convergence Check

A convergence subset compared 3000 and 7000 training iterations for two scenes and four splits: 5views_uniform, 5views_random, 5views_quality_image_diverse, and 20views_uniform. The 7000-iteration runs did not improve held-out PSNR in this subset (Fig. 6). The mean PSNR change from 3000 to 7000 iterations was -2.531 dB, with all eight measured cases showing lower PSNR at 7000 iterations. This does not establish that 3000 iterations are optimal in every setting; rather, it supports using 3000 iterations as a consistent comparative setting for the current benchmark.

Foundation-Model Pose Proxy Benchmark

VGGT, DUSt3R, and MASt3R were evaluated using a sparse-view camera-pose proxy metric [8-10]. Predicted camera centers were similarity-aligned to the full-scene COLMAP reference trajectory and evaluated by normalized camera-center RMSE. VGGT achieved the lowest mean aligned camera-center RMSE in this proxy benchmark. These results should not be interpreted as full novel-view synthesis results for VGGT, DUSt3R, or MASt3R unless downstream rendering is added.

The pose proxy is nevertheless useful because camera geometry is a prerequisite for many downstream reconstruction and rendering workflows. If a foundation model predicts camera centers that are more consistent with a reference trajectory, that can indicate better geometric organization under sparse input. However, pose consistency alone does not measure texture fidelity, occlusion handling, view-dependent effects, or the quality of rendered novel views. The proxy results are therefore reported alongside, not merged with, the 3DGS rendering metrics.

InstantSplat Rendering Study

InstantSplat was evaluated in a limited 5-view rendering study across the same four scenes and four selection strategies [11]. Each run used MASt3R geometry initialization, 500 Gaussian training iterations, and 20 held-out test views. All 16 runs completed successfully. InstantSplat preserved coarse structure in some held-out views but produced visible smearing, ghosting, and incomplete geometry in several sparse-view cases. The evaluation uses a similarity-aligned test-camera bridge and should be treated as limited evidence rather than a definitive method ranking.

Discussion

Based on the results, three practical lessons for sparse-view 3DGS benchmarking are proposed. First, image quality alone is not enough: selecting only sharp, well-exposed, high-feature images can focus views on similar parts of the scene and therefore limit geometric coverage. Second, diversity is important, but pose-aware diversity requires camera estimates that may not be available before reconstruction. Third, random and uniform baselines are still strong and should not be considered weak baselines.

These lessons are directly applicable to reproducible assessment. A method evaluated only with a quality-only selector may seem better than it really is, since being evaluated with a quality-only selector may be a poor baseline in an environment with sparse data. On the other hand, a selector which can be improved by quality-only selection cannot be said to be generally superior if not compared to repeated random seeds and uniform spacing. The benchmark thus distinguishes between the practical question of whether a Selector might be used prior to reconstruction, and the diagnostic question of how much performance might be realized if camera-center diversity was known.

It is important to report exposure shifts explicitly in degradation experiments. For this controlled benchmark, severe underexposure resulted in higher average losses than JPEG compression, motion blur or the simulated low-resolution case tested. These results isolate training-input degradation as the data was clean from held-out test images, and not a combination of training and evaluation corruption.

The outcome of the exposure is also important for smartphone and field capture. Errors in the exposure can lead to decreased feature visibility, texture suppression and alter the appearance statistics observed during training by the Gaussian optimization. Compression and blur are still significant but may be dependent on the content of the scene, camera movements, and the downsampling pipeline. The present degradation block is thus to be seen as a controlled first step that detects sensitivity patterns in the assessed 5-view setting and suggests scripts for extending the same protocol to further views, scenes and combined degradations.

The foundation-model results are helpful; however, they are restricted. The VGGT, DUSt3R, and MASt3R results are pose-proxy evidence, not rendered novel-view quality. InstantSplat is closer to a rendering comparison, but the current experiment has 5-view inputs, 500 training iterations, and an aligned-test-camera bridge. More comprehensive method-native rendering evaluation is required before more confident method-ranking claims can be made.

The study also explains how to include foundation-model-based methods in benchmark papers. Although feed forward or pose-free models can be quick to build and geometrically realistic, the output of these models must be assessed using standards that are commensurate with the claim being made. In the case of pose consistency, Camera-center RMSE would support a statement; in the case of novel-view synthesis, PSNR, SSIM, and LPIPS would support a statement; in the case of reconstructed shape with ground truth, geometry metrics like Chamfer distance or F-score would be required. These evidence types are kept apart to minimize the chance of inappropriate comparisons.

Methods

Datasets

The benchmark uses four public scenes from two datasets: Tanks and Temples train and truck, and Deep Blending drjohnson and playroom [3,4]. Each scene includes images and a COLMAP sparse reconstruction. For each scene, 20 held-out test views were selected and excluded from all sparse training splits.

No additional Mip-NeRF 360, DTU, LLFF, or official 3DGS-compatible public scenes were available locally during the final polishing phase. The repository therefore includes a supplementary smartphone-scene protocol for external validation. New smartphone scenes should follow the same folder conventions, include metadata CSV files, reserve clean held-out test views before sparse-view selection, and then run the existing split, selection, degradation, rendering, and evaluation scripts without changing the train/test separation rule.

Raw public dataset images are not redistributed in the lightweight submission package. Users should obtain Tanks and Temples and Deep Blending data from official sources and then use the provided scripts to reproduce split generation, training, rendering, and analysis. Generated CSV summaries, split metadata, figures, and manuscript tables may be released with the code package, subject to final license verification for any dataset-derived assets.

View Selection

Five selection strategies were evaluated. uniform selects evenly spaced images from the image sequence. random selects deterministic random samples from candidate training images. quality ranks images using sharpness, exposure, entropy, and ORB feature count. quality_diverse greedily combines image quality with COLMAP camera-center diversity and is therefore pose-aware. quality_image_diverse greedily combines image quality, color histogram distance, and ORB descriptor matching distance without using camera positions.

Gaussian Splatting

Each selected split was materialized into a COLMAP-compatible scene folder containing selected training images, camera metadata, image metadata, and filtered sparse points. The primary 3DGS benchmark used 3000 training iterations with resolution scale 4 [1]. Evaluation scenes included the same training cameras plus 20 held-out cameras, with test.txt used to enforce test-view separation in rendering.

Degradation Protocol

For degradation experiments, only training images were degraded. Held-out test images remained clean. Completed degradation conditions were JPEG Q20, JPEG Q40, motion blur kernel 15, low-resolution 50%, underexposure 0.6, underexposure 0.4, and overexposure 1.4. Degradation summaries report both raw PSNR/SSIM/LPIPS and changes relative to the matched clean 5-view baseline.

Foundation-Model Pose Benchmark

VGGT, DUSt3R, and MASt3R were run on sparse image sets using official checkpoints [8-10]. Predicted camera centers were extracted, similarity-aligned to full-scene COLMAP camera centers, and evaluated with normalized RMSE. Runtime and peak GPU memory were recorded.

InstantSplat Evaluation

InstantSplat was run on 5-view splits with MASt3R geometry initialization and 500 Gaussian training iterations [10,11]. Full-scene COLMAP test cameras were aligned into the InstantSplat coordinate frame using a similarity transform estimated from the five training camera centers. Rendered test views were compared with clean held-out images using PSNR, SSIM, and LPIPS.

Statistical Analysis

Paired statistical comparisons were computed using matched scene, view-count, and held-out image pairs. Mean differences were reported with paired bootstrap confidence intervals and Wilcoxon signed-rank p-values [12,13]. For PSNR and SSIM, positive differences favor the target method. For LPIPS, negative differences favor the target method.

Use of AI-assisted tools

AI-assisted tools were used for code organization, drafting support, and language polishing. All experimental design decisions, result interpretation, manuscript claims, and final content were reviewed and approved by the author(s). AI tools were not listed as authors.

Limitations

First, pose-aware quality-diverse selection uses full-scene COLMAP camera centers, so it is not a deployable pre-reconstruction selector. It is included as a diagnostic baseline. The pose-free image-diverse selector is more practical but weaker than random and uniform baselines in the current results.

Second, VGGT, DUSt3R, and MASt3R are evaluated with pose proxy metrics, not full rendered novel-view metrics. Their results should not be directly compared to 3DGS PSNR/SSIM/LPIPS results unless downstream rendering is added.

Third, the InstantSplat benchmark is a limited 5-view study with 500 training iterations and an aligned-test-camera evaluation bridge. Longer training and method-native evaluation would be needed for a more complete InstantSplat comparison.

Fourth, the degradation block covers one 5-view setting. It includes multiple degradation severities, but it does not test every view count or every combined degradation.

Fifth, dataset coverage is limited to four scenes from two public datasets. Additional dataset families and smartphone scenes are planned as external validation, so the current findings should be interpreted as controlled benchmark evidence rather than broad performance conclusions.

Sixth, the current work emphasizes image-space metrics and pose proxies. Geometry metrics such as Chamfer distance and F-score would require datasets with verified ground-truth geometry and consistent evaluation masks. Adding such geometry evaluation on DTU or Tanks and Temples ground-truth subsets would strengthen future versions of the benchmark.

Data Availability

The processed result tables, sparse-view split metadata, evaluation summaries, and manuscript figures generated for this study are available at https://doi.org/10.5281/zenodo.20478630. Raw Tanks and Temples and Deep Blending images are not redistributed with this release and should be obtained from the official dataset providers subject to their terms of use.

Code Availability

The custom scripts used for sparse-view split generation, image-quality scoring, degradation synthesis, 3DGS evaluation summarization, statistical testing, and manuscript table generation are available at https://github.com/XessX/sparse3d-gaussian-splatting-benchmark and archived at https://doi.org/10.5281/zenodo.20478630. Third-party methods, including COLMAP, GraphDeco 3D Gaussian Splatting, VGGT, DUSt3R, MASt3R, and InstantSplat, should be obtained from their official repositories and used under their respective licenses.

Acknowledgements

The author received no specific funding for this work.

Author Contributions

Al Jubair Hossain: conceptualization, methodology, software, validation, formal analysis, investigation, data curation, writing - original draft, writing - review and editing, visualization, and project administration.

Competing Interests

The author declares no competing interests.

Supplementary Information

Supplementary Information includes additional figures, tables, result summaries, and the smartphone-scene external-validation protocol. The supplementary smartphone protocol is cited as Supplementary Note S1. Extra result curves and qualitative grids are listed as Supplementary Figs. S1-S26, and additional CSV-derived tables are listed as Supplementary Tables S1-S17.

Figure Legends

Fig. 1. Benchmark workflow. Public scenes are split into held-out test views and sparse training views before selection. Degradation experiments alter training images only, while held-out test views remain clean.

Fig. 2. Clean sparse-view 3DGS performance. Mean held-out PSNR across view counts and view-selection strategies for the four evaluated scenes.

Fig. 3. Repeated random-seed variability. Mean and standard deviation of held-out PSNR across five random seeds, four scenes, and six view counts.

Fig. 4. Five-view selector comparison. Mean held-out PSNR for quality-only, pose-free image-diverse, pose-aware quality-diverse, random, and uniform 5-view selection.

Fig. 5. Degradation sensitivity. PSNR change relative to clean 5-view baselines for JPEG compression, blur, low-resolution simulation, underexposure, and overexposure.

Fig. 6. Convergence subset. Held-out PSNR comparison for 3000 and 7000 3DGS training iterations on selected scenes and splits.

Table Legends

Table 1. Dataset summary. Public scene sources, image counts, held-out test-view counts, split counts, and dataset license-status notes.

Table 2. Main paired statistical tests. Paired comparisons for pose-free quality-image-diverse selection against quality-only, pose-aware quality-diverse, random, and uniform baselines.

References

1. Kerbl, B., Kopanas, G., Leimkuehler, T. & Drettakis, G. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics 42, 139 (2023). https://arxiv.org/abs/2308.04079

2. Schoenberger, J. L. & Frahm, J.-M. Structure-from-Motion Revisited. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 4104-4113 (2016). https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html

3. Knapitsch, A., Park, J., Zhou, Q.-Y. & Koltun, V. Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction. ACM Transactions on Graphics 36, 78 (2017). https://www.tanksandtemples.org/

4. Hedman, P., Philip, J., Price, T., Frahm, J.-M., Drettakis, G. & Brostow, G. Deep Blending for Free-Viewpoint Image-Based Rendering. ACM Transactions on Graphics 37, 257 (2018). https://discovery.ucl.ac.uk/id/eprint/10117776/

5. Huynh-Thu, Q. & Ghanbari, M. Scope of validity of PSNR in image/video quality assessment. Electronics Letters 44, 800-801 (2008). https://doi.org/10.1049/el:20080522

6. Wang, Z., Bovik, A. C., Sheikh, H. R. & Simoncelli, E. P. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing 13, 600-612 (2004). https://doi.org/10.1109/TIP.2003.819861

7. Zhang, R., Isola, P., Efros, A. A., Shechtman, E. & Wang, O. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 586-595 (2018). https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html

8. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C. & Novotny, D. VGGT: Visual Geometry Grounded Transformer. arXiv:2503.11651 (2025). https://arxiv.org/abs/2503.11651

9. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B. & Revaud, J. DUSt3R: Geometric 3D Vision Made Easy. arXiv:2312.14132 (2023). https://arxiv.org/abs/2312.14132

10. Leroy, V., Cabon, Y. & Revaud, J. Grounding Image Matching in 3D with MASt3R. arXiv:2406.09756 (2024). https://arxiv.org/abs/2406.09756

11. Fan, Z. et al. InstantSplat: Sparse-view SfM-free Gaussian Splatting in Seconds. arXiv:2403.20309 (2024). https://arxiv.org/abs/2403.20309

12. Wilcoxon, F. Individual Comparisons by Ranking Methods. Biometrics Bulletin 1, 80-83 (1945). https://doi.org/10.2307/3001968

13. Efron, B. & Tibshirani, R. J. An Introduction to the Bootstrap. Chapman & Hall/CRC (1993).

14. Mildenhall, B. et al. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In European Conference on Computer Vision 405-421 (2020). https://arxiv.org/abs/2003.08934

15. Mildenhall, B. et al. Local Light Field Fusion: Practical View Synthesis with Prescriptive Sampling Guidelines. ACM Transactions on Graphics 38, 1-14 (2019). https://bmild.github.io/llff/

16. Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P. & Hedman, P. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 5470-5479 (2022). https://openaccess.thecvf.com/content/CVPR2022/html/Barron_Mip-NeRF_360_Unbounded_Anti-Aliased_Neural_Radiance_Fields_CVPR_2022_paper.html

17. Jain, A., Tancik, M. & Abbeel, P. Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision 5885-5894 (2021). https://arxiv.org/abs/2104.00677
