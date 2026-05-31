from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_metric_vs_views(csv_path: str | Path, metric: str, output_path: str | Path):
    df = pd.read_csv(csv_path)
    if "views" not in df.columns or metric not in df.columns:
        raise ValueError(f"CSV must contain 'views' and '{metric}' columns")

    plt.figure(figsize=(7, 5))
    for method, group in df.groupby("method"):
        group = group.sort_values("views")
        plt.plot(group["views"], group[metric], marker="o", label=method)

    plt.xlabel("Number of input views")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} across sparse-view settings")
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
