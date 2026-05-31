# Submission Readiness Report

Project: **Benchmarking Gaussian Splatting Under Sparse and Degraded Views**

## What Is Ready

- Working CUDA/PyTorch/COLMAP/3DGS/foundation-model environment.
- Four-scene clean COLMAP + 3DGS benchmark with 120 primary held-out evaluations.
- Repeated random-seed baseline with 120 additional evaluated runs.
- Pose-free `quality_image_diverse` selector implemented and evaluated.
- Pose-aware `quality_diverse` retained as a diagnostic baseline.
- Completed 5-view degradation block with seven degraded training-image conditions and clean held-out test images.
- VGGT, DUSt3R, and MASt3R pose proxy benchmarks.
- InstantSplat 5-view held-out rendering study.
- Paired statistical tests for clean 3DGS per-view results.
- Convergence subset comparing 3000 vs 7000 iterations.
- Dataset summary, manuscript tables, figures, setup report, and audit report.
- Final Markdown, DOCX, and PDF manuscript exports.
- Final supplementary Markdown and PDF export.
- Polished cover letter and APC waiver request drafts.

## What Remains

- Add another public dataset or more scenes if time/storage allow.
- Add final affiliation, ORCID, public data/code repository links, and reviewer details.
- Decide whether foundation-model baselines stay as proxy metrics or receive full downstream rendering evaluation in a future expansion.
- Verify dataset license/redistribution terms before public release.
- Manually inspect the generated DOCX in Word or LibreOffice because automated visual rendering was unavailable in this environment.
- Optionally expand degradation beyond the 5-view setting or add combined degradations.

## Current Submission Readiness

Current status: **ready for human/advisor review, but not ready for direct Scientific Reports upload until manual author, repository, license, and portal-format fields are completed**.

The project is strong enough for an internal pre-submission draft and advisor review. The random-seed variance, expanded degradation severity work, manuscript expansion, supplementary information, cover letter, APC waiver request, and lightweight submission package are complete. The main remaining scientific weakness is broader dataset coverage, plus the fact that foundation-model baselines are still partly proxy evaluations.

## Strongest Next Runs

1. Add Mip-NeRF 360, DTU, LLFF, or more official public scenes if storage/download time permits.
2. Add method-native or downstream rendering evaluation for VGGT/DUSt3R/MASt3R if feasible.
3. Normalize path fields in CSVs before public release.
4. Optionally test degradation beyond 5 views.

## Bottom Line

The benchmark now supports the central cautious claim: quality-only sparse-view selection is weak, view diversity matters, and practical pose-free image diversity improves over quality-only selection but does not beat random, uniform, or pose-aware diversity in the current results.
