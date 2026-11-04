# Download the initial experiment data

The [initial experiments release](https://github.com/BshKatrin/tuxkart-dagger-rlpd/releases/tag/initial-experiments-v1)
contains a frozen snapshot of all 37 completed runs. Artifacts are release
attachments, separate from Git history. This repository is private, so you need
repository access to download them.

- **initial-experiments-data.zip:** demonstration datasets, configurations,
  evaluation and training logs, final trajectories, expert references, and the
  original training source snapshot. Enough for analyzing the experiments.
- **initial-experiments-checkpoints.zip:** full saved training checkpoints,
  including policies, optimizer state, and replay or aggregated imitation data.
  Download this too to use the live policy viewer.
- **SHA256SUMS:** checksums for both archives.
- **artifact-manifest.json:** archive contents, sizes, and packaging provenance.

Pilot runs and generated videos are excluded. Recordings can be recreated with
the [viewer](visualization.md).

## Restore into a fresh clone

With GitHub CLI authenticated to an account with repository access:

```sh
uv sync --locked
gh release download initial-experiments-v1 \
  --repo BshKatrin/tuxkart-dagger-rlpd \
  --dir dist/initial-experiments \
  --pattern '*.zip' --pattern SHA256SUMS --pattern artifact-manifest.json
(cd dist/initial-experiments && shasum -a 256 -c SHA256SUMS)
unzip -n dist/initial-experiments/initial-experiments-data.zip
unzip -n dist/initial-experiments/initial-experiments-checkpoints.zip
```

The archives restore `data/` and `runs/study/` under the project root. Both are
Git-ignored. Historical absolute paths and process IDs in original provenance
remain records of the original machine, not instructions for your machine.

Inspect the data with:

```sh
uv run python scripts/status.py runs/study
uv run python scripts/summarize.py runs/study
uv run python -m kartstudy.playback
```

Playback runs on CPU and requires graphical simulator support; training uses
Metal. These are full checkpoints, but automated training resume is not
implemented. The experimental protocol is in [initial-experiments.md](initial-experiments.md).

## Package a completed suite

```sh
uv run python scripts/package_results.py --output dist/initial-experiments
```

The packager checks completion, preserves the original files, verifies ZIP
integrity, and writes checksums. Use a fresh output directory for each snapshot.
Keep release archives immutable; publish a new version when the shared data changes.
