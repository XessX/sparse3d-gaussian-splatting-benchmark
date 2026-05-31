# Benchmarking Gaussian Splatting Under Sparse and Degraded Views

This public-release folder contains cleaned scripts, result summaries, manuscript tables, selected figures, and reproducibility metadata for a Scientific Reports-style benchmark of sparse-view 3D Gaussian Splatting under degraded image conditions.

## Summary

The benchmark evaluates sparse-view reconstruction behavior under limited views, image degradation, and different view-selection strategies. It includes:

- COLMAP + 3D Gaussian Splatting summaries;
- repeated random-seed baselines;
- the pose-free `quality_image_diverse` selector;
- degradation analysis with clean held-out test views;
- foundation-model pose proxy summaries for VGGT, DUSt3R, and MASt3R;
- a limited InstantSplat 5-view rendering study.

## Dataset Sources

The experiments use public Tanks and Temples and Deep Blending scenes as distributed through official sources or the authorized 3D Gaussian Splatting dataset package. Raw dataset images are not redistributed in this release. Users must download public datasets from official providers and follow the original terms of use.

## Environment

```bash
conda env create -f environment.yml
conda activate sparse3d
```

or:

```bash
python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
```

## Reproduction

The repository includes modular scripts for sparse-view split generation, view selection, degradation synthesis, 3DGS evaluation summarization, foundation-model pose proxy collection, statistical testing, and table/figure generation.

Representative commands are listed in the main `README_original.md` copied from the working project. Heavy external methods and raw datasets are not included; obtain them from their official repositories and dataset providers.

## Expected Outputs

Important generated outputs include:

- `results/csv/`
- `manuscript/tables/`
- `manuscript/figures/`
- `results/logs/setup_report.md`

## Citation

See `CITATION.cff`.

- GitHub repository: https://github.com/XessX/sparse3d-gaussian-splatting-benchmark
- GitHub release: https://github.com/XessX/sparse3d-gaussian-splatting-benchmark/releases/tag/v0.1.5
- Zenodo record: https://zenodo.org/records/20477399
- Zenodo DOI: https://doi.org/10.5281/zenodo.20477399

## License

License selection requires author confirmation. See `LICENSE_PENDING.md`. Third-party datasets and methods are governed by their own licenses.

## Contact

Al Jubair Hossain  
American International University-Bangladesh (AIUB)  
jubair.hossain@aiub.edu  
ORCID: 0009-0005-3498-5826
