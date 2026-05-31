# Table: Paired statistical tests for quality-image-diverse selection

Differences are target minus comparator. For PSNR/SSIM, positive favors the target; for LPIPS, negative favors the target.

| target | comparator | metric | n_pairs | mean_diff_target_minus_comparator | bootstrap_ci_low | bootstrap_ci_high | wilcoxon_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| quality_image_diverse | quality | psnr | 480 | 1.134 | 0.8663 | 1.406 | 1.643e-14 |
| quality_image_diverse | quality | ssim | 480 | 0.066 | 0.05206 | 0.08028 | 1.265e-15 |
| quality_image_diverse | quality | lpips | 480 | -0.04235 | -0.05104 | -0.03385 | 3.48e-18 |
| quality_image_diverse | quality_diverse | psnr | 480 | -1.634 | -1.909 | -1.353 | 1.395e-24 |
| quality_image_diverse | quality_diverse | ssim | 480 | -0.06029 | -0.07457 | -0.04547 | 3.329e-14 |
| quality_image_diverse | quality_diverse | lpips | 480 | 0.04571 | 0.03513 | 0.05639 | 4.058e-16 |
| quality_image_diverse | random | psnr | 480 | -2.251 | -2.722 | -1.795 | 1.032e-18 |
| quality_image_diverse | random | ssim | 480 | -0.07896 | -0.1008 | -0.05751 | 6.486e-11 |
| quality_image_diverse | random | lpips | 480 | 0.05936 | 0.04386 | 0.07529 | 3.5e-12 |
| quality_image_diverse | uniform | psnr | 480 | -1.84 | -2.249 | -1.443 | 5.082e-19 |
| quality_image_diverse | uniform | ssim | 480 | -0.07539 | -0.09471 | -0.05565 | 5.412e-12 |
| quality_image_diverse | uniform | lpips | 480 | 0.04872 | 0.03489 | 0.06255 | 1.693e-10 |

