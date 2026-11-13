"""Package a completed study for GitHub Releases, keeping artifacts out of Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kartstudy.environment import SCHEMA_VERSION


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_archive(path, files):
    with zipfile.ZipFile(
        path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3
    ) as archive:
        for source in files:
            archive.write(
                source,
                arcname=source.as_posix(),
                compress_type=zipfile.ZIP_STORED
                if source.suffix in [".npz", ".gz"]
                else zipfile.ZIP_DEFLATED,
            )
    with zipfile.ZipFile(path) as archive:
        bad_file = archive.testzip()
        if bad_file:
            raise RuntimeError(f"ZIP integrity failure: {bad_file}")
    if path.stat().st_size >= 2 * 1024**3:
        raise ValueError(f"{path} exceeds the GitHub Release asset limit")
    result = {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "files": len(files),
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/study"))
    parser.add_argument("--root", type=Path, default=Path("runs/study-polar-four"))
    parser.add_argument("--monte-carlo", action="store_true", help="Package the completed five-case Monte Carlo diagnostic")
    args = parser.parse_args()
    if args.monte_carlo:
        package_monte_carlo(args.output)
        return
    root = args.root
    manifest = json.loads((root / "manifest.json").read_text())
    selection = json.loads((root / "selected-policies.json").read_text())
    if set(selection) not in (
        {"bc", "dagger", "sac", "rlpd"},
        {"bc", "dagger", "sac", "sac_sym", "rlpd"},
    ):
        raise ValueError("Select exactly one completed policy per method")
    if manifest["state"] not in {"stopped", "completed"}:
        raise ValueError("Stop the suite before packaging")
    datasets, retained, originals, extras = set(), [], [], []
    inventory = []
    jobs = list(manifest["jobs"])
    known = {Path(job["output"]).resolve() for job in jobs}
    for value in selection.values():
        run = (root / value).resolve()
        if run not in known:
            jobs.append({"output": str(run.relative_to(Path.cwd()))})
    for job in jobs:
        run = Path(job["output"])
        status_path = run / "status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        config = json.loads((run / "config.json").read_text())
        chosen = (
            config["algo"] in selection
            and (root / selection[config["algo"]]).resolve() == run.resolve()
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Incompatible suite schema")
        if chosen and (
            status["status"] != "completed"
            or config["seed"] != 0
            or status["step"] != (0 if config["algo"] == "bc" else config["steps"])
        ):
            raise ValueError(
                f"Retained policy must be complete, training seed 0: {run}"
            )
        inventory.append(
            {
                "run": run.name,
                "retained": chosen,
                "status": status["status"],
                "seed": config["seed"],
            }
        )
        files = [
            p
            for p in sorted(run.glob("*"))
            if p.is_file() and p.suffix in {".json", ".jsonl", ".npz"}
        ]
        if chosen:
            import torch

            from kartstudy.playback import load_actor

            actor, actor_config, step = load_actor(run)
            actor_path = run / "actor.pt"
            if not actor_path.exists():
                torch.save(
                    {"actor": actor.state_dict(), "config": actor_config, "step": step},
                    actor_path,
                )
            retained.extend(files + [actor_path])
            originals.append(run / "checkpoint.pt")
            if run.parent.resolve() != root.resolve():
                retained.extend(
                    p
                    for p in run.parent.glob("*")
                    if p.is_file()
                    and p.name in {"source.tar.gz", "comparison.json", "config.json"}
                )
            dataset = Path(config["dataset"])
            datasets.update([dataset, dataset.with_suffix(".json")])
        else:
            extras.extend(
                files
                + ([run / "checkpoint.pt"] if (run / "checkpoint.pt").exists() else [])
            )
    if sum(row["retained"] for row in inventory) != len(selection):
        raise ValueError("Missing selected policies")
    metadata = [
        root / name
        for name in [
            "manifest.json",
            "source.tar.gz",
            "qualification.json",
            "selected-policies.json",
        ]
    ]
    metadata += sorted((root / "configs").glob("*.json"))
    diagnostics = [
        p
        for directory in ["runs/expert-road-quality-final", "runs/expert-all-tracks"]
        for p in Path(directory).glob("*")
        if p.suffix in {".json", ".csv"}
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    groups = [
        ("selected-policies.zip", retained + [root / "selected-policies.json"]),
        ("study-data.zip", sorted(datasets) + metadata + diagnostics),
        ("selected-training-checkpoints.zip", originals),
        ("extra-policies.zip", extras),
    ]
    if any((args.output / name).exists() for name, _ in groups):
        raise FileExistsError(
            "Use a fresh output directory; archives are never overwritten"
        )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(create_archive, args.output / name, files)
            for name, files in groups
        ]
        assets = [future.result() for future in futures]
    checksums = "".join(f"{asset['sha256']}  {asset['name']}\n" for asset in assets)
    (args.output / "SHA256SUMS").write_text(checksums)
    description = {
        "suite": str(root),
        "retained_training_seed": 0,
        "runs": inventory,
        "packaging_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "packaging_script_sha256": sha256(Path(__file__)),
        "assets": assets,
        "extraction": "Extract selected-policies.zip and study-data.zip at the repository root. Other ZIPs are optional.",
        "included": "Selected seed-0 policies, optional extra and stopped runs, demonstrations, expert diagnostics, original training source snapshot and full checkpoints.",
        "excluded": "Pilot runs, rendered videos, operational launch/PID files and the virtual environment.",
    }
    (args.output / "artifact-manifest.json").write_text(
        json.dumps(description, indent=2) + "\n"
    )
    print("Verified release ZIP archives and wrote SHA256SUMS.", flush=True)


def package_monte_carlo(output):
    from kartstudy.critic_artifacts import prepare_release, verify

    output.mkdir(parents=True, exist_ok=False)
    groups = prepare_release(Path.cwd())
    validation = verify(Path.cwd())
    assets = [create_archive(output / name, files) for name, files in groups]
    manifest = {
        "packaging_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "packaging_script_sha256": sha256(Path(__file__)),
        "validation": validation, "assets": assets,
        "extraction": "Verify SHA256SUMS; extract both ZIPs at the repository root, preserving paths.",
        "frozen_models": "Derived evaluation-only checkpoints retain exact model weights, temperature and configuration. Original full-checkpoint hashes are preserved; optimizers and replay are excluded.",
        "verification": "uv run python -m kartstudy.critic_artifacts",
    }
    path = output / "artifact-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    checksums = [(asset["sha256"], asset["name"]) for asset in assets] + [(sha256(path), path.name)]
    (output / "SHA256SUMS").write_text("".join(f"{checksum}  {name}\n" for checksum, name in checksums))
    print(json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
