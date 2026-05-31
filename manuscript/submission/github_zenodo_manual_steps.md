# GitHub, Zenodo, and OSF Manual Steps

These steps are required because GitHub CLI is not installed in the current environment and Windows Computer Use could not connect to the Chrome automation helper during this session. No GitHub repository, Zenodo DOI, or OSF record has been fabricated.

## GitHub Repository

1. Go to GitHub and log in.
2. Create a new public repository named `sparse3d-gaussian-splatting-benchmark`.
3. Add this description:
   `Sparse-view 3D Gaussian Splatting benchmark with degradation tests, view-selection baselines, result summaries, and Scientific Reports reproducibility materials.`
4. Do not initialize the repository with a README, license, or `.gitignore` if you plan to push the prepared local folder.
5. From the prepared folder `public_release_sparse3d_scirep`, run:

```powershell
git init
git add .
git commit -m "Initial public benchmark release"
git branch -M main
git remote add origin https://github.com/<OWNER>/sparse3d-gaussian-splatting-benchmark.git
git push -u origin main
git tag v0.1.0
git push origin v0.1.0
```

6. Create a GitHub release from tag `v0.1.0`.
7. Use release title:
   `Sparse3D Gaussian Splatting Scientific Reports Benchmark v0.1.0`
8. Use release notes:

```text
This release contains cleaned scripts, result summaries, manuscript tables, selected figures, and reproducibility metadata for the Scientific Reports submission.

Raw public dataset images are not redistributed.
External method repositories and datasets must be obtained from their official sources and used under their respective licenses.
```

9. Copy the final repository URL and release URL into:
   - `CITATION.cff`
   - `manuscript/submission/data_availability_statement.md`
   - `manuscript/submission/code_availability_statement.md`
   - `manuscript/submission/scientific_reports_manuscript_final.md`
   - `SCIENTIFIC_REPORTS_FINAL_CHECKLIST.md`

## Zenodo DOI

1. Go to Zenodo and log in.
2. Connect Zenodo to GitHub if it is not already connected.
3. In Zenodo GitHub settings, enable the repository `sparse3d-gaussian-splatting-benchmark`.
4. If the GitHub release already exists, re-run or refresh archive creation if Zenodo offers that option.
5. Otherwise, create or re-create GitHub release `v0.1.0`.
6. Wait for Zenodo to archive the release.
7. Copy the generated version DOI and concept DOI.
8. Paste the DOI into:
   - `CITATION.cff`
   - `manuscript/submission/data_availability_statement.md`
   - `manuscript/submission/code_availability_statement.md`
   - `manuscript/submission/scientific_reports_manuscript_final.md`
   - `SCIENTIFIC_REPORTS_FINAL_CHECKLIST.md`

Do not upload raw Tanks and Temples or Deep Blending images to Zenodo unless redistribution terms are verified.

## OSF Fallback

If Zenodo cannot archive the release:

1. Create an OSF project for the benchmark.
2. Upload the cleaned public-release ZIP only.
3. Do not upload raw dataset images, external repositories, checkpoints, credentials, or local caches.
4. Record the OSF project link as a temporary archive link.
5. Prefer a Zenodo DOI if it becomes available later.
