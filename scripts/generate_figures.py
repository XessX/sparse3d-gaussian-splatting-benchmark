from pathlib import Path
import glob
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.plots import plot_metric_vs_views

def main():
    csv_files = glob.glob("results/csv/**/*placeholder_results.csv", recursive=True)
    if not csv_files:
        print("No placeholder result CSV found. Run scripts/run_baseline_placeholder.py first.")
        return

    for csv_path in csv_files:
        stem = Path(csv_path).stem
        for metric in ["psnr", "ssim", "lpips"]:
            out_path = Path("results/figures") / f"{stem}_{metric}.png"
            plot_metric_vs_views(csv_path, metric, out_path)
            print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
