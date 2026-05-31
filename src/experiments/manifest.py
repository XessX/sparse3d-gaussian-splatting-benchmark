from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    scene: str
    method: str
    views: int
    selection: str
    condition: str
    split_name: str
    input_image_dir: str
    output_dir: str
    status: str = "pending"


def make_experiment_id(parts: list[str]) -> str:
    digest = sha1("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


def resolve_input_dir(scene: str, condition: str) -> Path:
    if condition == "clean":
        return Path("datasets/raw") / scene / "images"
    return Path("datasets/processed") / scene / "degraded" / condition


def build_manifest(
    scene: str,
    methods: list[str],
    view_counts: list[int],
    selections: list[str],
    conditions: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for method in methods:
        for views in view_counts:
            for selection in selections:
                for condition in conditions:
                    split_name = f"{views}views_{selection}"
                    input_dir = resolve_input_dir(scene, condition)
                    exp_id = make_experiment_id(
                        [scene, method, str(views), selection, condition]
                    )
                    output_dir = (
                        Path("results")
                        / "method_outputs"
                        / method
                        / scene
                        / condition
                        / split_name
                    )
                    rows.append(
                        asdict(
                            Experiment(
                                experiment_id=exp_id,
                                scene=scene,
                                method=method,
                                views=views,
                                selection=selection,
                                condition=condition,
                                split_name=split_name,
                                input_image_dir=str(input_dir),
                                output_dir=str(output_dir),
                                status="pending" if input_dir.exists() else "missing_input",
                            )
                        )
                    )
    return rows
