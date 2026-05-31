import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir


METHOD_REPOS = {
    "gaussian_splatting": PROJECT_ROOT / "external_methods" / "gaussian-splatting",
    "vggt": PROJECT_ROOT / "external_methods" / "vggt",
    "dust3r": PROJECT_ROOT / "external_methods" / "dust3r",
    "mast3r": PROJECT_ROOT / "external_methods" / "mast3r",
    "instantsplat": PROJECT_ROOT / "external_methods" / "InstantSplat",
}


def git_value(repo: Path, args: list[str]) -> str | None:
    if not repo.exists():
        return None
    command = ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def main():
    records = []
    for name, repo in METHOD_REPOS.items():
        records.append(
            {
                "method": name,
                "local_path": str(repo),
                "exists": repo.exists(),
                "commit": git_value(repo, ["rev-parse", "HEAD"]),
                "short_commit": git_value(repo, ["rev-parse", "--short", "HEAD"]),
                "remote": git_value(repo, ["remote", "get-url", "origin"]),
            }
        )

    output = PROJECT_ROOT / "results" / "logs" / "external_methods.json"
    ensure_dir(output.parent)
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved external method records to {output}")
    for record in records:
        status = "ok" if record["exists"] else "missing"
        print(f"{record['method']}: {status} {record['short_commit'] or ''}")


if __name__ == "__main__":
    main()
