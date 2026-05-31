# Sparse3D Setup Report

## Environment
- Python: 3.10.9
- Platform: Windows-10-10.0.26200-SP0
- Torch: 2.11.0+cu128
- Torch CUDA available: True
- CUDA devices: NVIDIA GeForce RTX 4080 Laptop GPU
- Missing Python packages: none
- Missing executables: none
- NVCC path: nvcc

## External Source Repositories
- gaussian_splatting: present (54c035f)
- vggt: present (a288dd0)
- dust3r: present (4c24a6e)
- mast3r: present (f5209af)
- instantsplat: present (b951567)

## Import Status
- vggt: ok
- dust3r: ok
- mast3r: ok
- instantsplat: ok
- gaussian_splatting_rasterizer: ok
- simple_knn: ok
- fused_ssim: ok

## Smoke-Test Outputs
- COLMAP status: success
- COLMAP models: 1
- COLMAP runtime_sec: 0.738
- COLMAP output: results\method_outputs\demo_scene\colmap_sfm
- VGGT device: cuda
- VGGT inference_sec: 0.876
- VGGT peak_gpu_memory_mb: 7365.8
- VGGT output previews: 4
- Real-scene VGGT scene: tandt/train
- Real-scene VGGT inference_sec: 1.210
- Real-scene VGGT peak_gpu_memory_mb: 7023.3
- Real-scene VGGT previews: 4
- Real-scene 3DGS smoke status: success
- Real-scene 3DGS iterations: 50
- Real-scene 3DGS train PSNR@50: 13.051
- Real-scene 3DGS point cloud: results/method_outputs/tandt/train/3dgs_smoke/point_cloud/iteration_50/point_cloud.ply
- Sparse 3DGS pilot rows at 200 iterations: 18
- Sparse 3DGS train-only rows at 3000 iterations: 24
- Best 3000-iteration train PSNR: 2views_quality = 50.495
- Sparse 3DGS held-out evaluation rows: 24
- Best held-out PSNR: 20views_uniform = 17.140
- Multi-scene 3DGS held-out evaluation rows: 240
- Multi-scene scenes: db/drjohnson, db/playroom, tandt/train, tandt/truck
- Multi-scene selections: quality, quality_diverse, quality_image_diverse, random, random_seed0, random_seed1, random_seed2, random_seed3, random_seed4, uniform
- Best multi-scene held-out PSNR: db/playroom / 20views_random = 22.120
- Best mean held-out PSNR: 20 views / random = 19.611
- Degradation robustness rows including clean baselines: 128
- Degraded-input 3DGS runs: 112
- Best degraded mean PSNR: blur_k15 / quality_diverse = 14.296
- Worst degraded mean PSNR: under_0.4 / quality = 7.917
- Foundation 5-view pose rows: 48
- Foundation methods: DUSt3R, MASt3R, VGGT
- Foundation failed pose rows: 0
- Best 5-view foundation pose RMSE: VGGT / random = 0.047
- Best overall 5-view foundation pose RMSE: VGGT = 0.073
- VGGT sparse pose summary rows: 24
- Best VGGT 20-view pose RMSE: random = 0.013
- InstantSplat smoke status: success (init 18.897s, train 12.494s, point cloud saved: True)
- InstantSplat held-out rendering rows: 16
- InstantSplat scenes: db/drjohnson, db/playroom, tandt/train, tandt/truck
- InstantSplat failed rendering rows: 0
- Best InstantSplat PSNR: tandt/truck / 5views_uniform = 13.618
- Best mean InstantSplat PSNR: uniform = 11.471
- Repeated random-seed rows prepared: 120
- Repeated random-seed rows evaluated: 120
- 3DGS convergence rows: 16
- 3DGS convergence evaluated rows: 16
- Statistical test rows: 84
- All-view statistical comparison rows: 12

## Dataset Outputs
- `tandt/train` images: 301
- Degraded `tandt/train` smoke images: 20
- Planned experiment rows: 360
- Planned rows with missing inputs: 0

## Remaining Blocker
- None for the starter pipeline: CUDA, COLMAP, VGGT, DUSt3R, MASt3R, InstantSplat imports, and GraphDeco native extensions are ready.
