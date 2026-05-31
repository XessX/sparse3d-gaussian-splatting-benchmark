import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir


SCENES = ["tandt/train", "tandt/truck", "db/drjohnson", "db/playroom"]
SPLITS = ["5views_uniform", "5views_random", "5views_quality", "5views_quality_diverse"]
NEW_CONDITIONS = ["jpeg_q40", "under_0.4", "over_1.4"]


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["condition", "planned_runs", "evaluated_runs", "status", "todo"]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in df[columns].iterrows()]
    return "\n".join([header, divider, *rows])


def main() -> None:
    existing_path = PROJECT_ROOT / "results" / "csv" / "3dgs_degradation_runs_3000.csv"
    existing = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()
    rows = []
    for condition in NEW_CONDITIONS:
        planned = len(SCENES) * len(SPLITS)
        evaluated = 0
        if not existing.empty and "condition" in existing.columns:
            evaluated = int((existing["condition"] == condition).sum())
        rows.append(
            {
                "condition": condition,
                "planned_runs": planned,
                "evaluated_runs": evaluated,
                "status": "complete" if evaluated >= planned else "todo",
                "todo": "" if evaluated >= planned else "Run scripts/run_3dgs_degradation.py with this condition for all scenes/splits.",
            }
        )
    df = pd.DataFrame(rows)
    out_dir = ensure_dir(PROJECT_ROOT / "results" / "csv" / "degradation")
    df.to_csv(out_dir / "degradation_expansion_todo.csv", index=False)
    table_dir = ensure_dir(PROJECT_ROOT / "manuscript" / "tables")
    df.to_csv(table_dir / "table_degradation_expansion_todo.csv", index=False)
    completed = int((df["status"] == "complete").sum())
    if completed == len(df):
        note = "All added degradation expansion conditions have been evaluated for the planned scenes and splits."
    else:
        note = "Some added degradation expansion conditions remain to be evaluated."
    (table_dir / "table_degradation_expansion_todo.md").write_text(
        "# Table: Additional degradation conditions\n\n"
        f"{note}\n\n"
        + markdown_table(df)
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved degradation expansion TODO table to {table_dir / 'table_degradation_expansion_todo.md'}")


if __name__ == "__main__":
    main()
