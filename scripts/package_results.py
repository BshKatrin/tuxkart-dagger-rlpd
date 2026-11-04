"""Package a completed study for GitHub Releases, keeping artifacts out of Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


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
    parser.add_argument("--output", type=Path, default=Path("dist/initial-experiments"))
    args = parser.parse_args()
    root = Path("runs/study")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["state"] != "completed" or any(
        j["state"] != "completed" for j in manifest["jobs"]
    ):
        raise ValueError(
            "Only completed suites can be published as experiment snapshots"
        )
    for job in manifest["jobs"]:
        run = Path(job["output"])
        status = json.loads((run / "status.json").read_text())
        expected = 0 if job["algo"] == "bc" else job["steps"]
        if (
            status["status"] != "completed"
            or status["step"] != expected
            or not (run / "checkpoint.pt").exists()
        ):
            raise ValueError(f"Incomplete run: {run}")
    # An explicit allowlist excludes pilots, videos, live viewer state and unrelated local files.
    datasets = sorted(
        p for p in Path("data").glob("*") if p.suffix in [".npz", ".json"]
    )
    metadata = [
        root / name for name in ["manifest.json", "hardware.json", "source.tar.gz"]
    ]
    qualification = Path("runs/expert_qualification.json")
    if qualification.exists():
        metadata.append(qualification)
    run_files = []
    checkpoints = []
    run_directories = [Path(j["output"]) for j in manifest["jobs"]]
    run_directories += [
        root / name for name in ["expert_reference", "expert_conservative"]
    ]
    for run in run_directories:
        for source in sorted(run.glob("*")):
            if not source.is_file():
                continue
            if source.name == "checkpoint.pt":
                checkpoints.append(source)
            elif source.suffix in [".json", ".jsonl", ".npz"]:
                run_files.append(source)
    args.output.mkdir(parents=True, exist_ok=True)
    groups = [
        ("initial-experiments-data.zip", datasets + metadata + run_files),
        ("initial-experiments-checkpoints.zip", checkpoints),
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
        "suite": "initial-experiments",
        "completed_runs": len(manifest["jobs"]),
        "packaging_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "packaging_script_sha256": sha256(Path(__file__)),
        "assets": assets,
        "extraction": "Extract both ZIPs at the repository root; their paths are complementary.",
        "included": "Demonstration datasets, all completed study runs, expert references and qualification, original training source snapshot, full checkpoints.",
        "excluded": "Pilot runs, rendered videos, operational launch/PID files and the virtual environment.",
    }
    (args.output / "artifact-manifest.json").write_text(
        json.dumps(description, indent=2) + "\n"
    )
    print("Verified both ZIP archives and wrote SHA256SUMS.", flush=True)


if __name__ == "__main__":
    main()
