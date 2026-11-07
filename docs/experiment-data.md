# Experiment artifacts

Use the latest **polar-four** release. Retain **one policy per method, training
seed 0**, consistently across collaborators. The old `initial-experiments-v1`
release uses the superseded setup and is not compatible with this study.

Extract ZIPs at the repository root, preserving their paths:

| Archive | Contents | Needed for playback? |
|---|---|---|
| `selected-policies.zip` | Four seed-0 actors, configs, statuses, evaluation logs and trajectories; policy selection | Yes |
| `study-data.zip` | Demonstrations and metadata, original suite/source/qualification, configurations and expert diagnostics | Yes |
| `selected-training-checkpoints.zip` | Original full seed-0 checkpoints, including optimizer/replay | No |
| `extra-policies.zip` | Other training seeds, including stopped runs, with original files | No |

The small `actor.pt` files contain only actor weights, configuration and step.
Playback prefers these and can fall back to full `checkpoint.pt` files. Full
checkpoints are preserved without modification. Stopped extra runs are explicitly
marked in their `status.json` and release inventory; they are not completed
experiments. The suite manifest preserves its original stopped queue state.

Verify `SHA256SUMS` before extraction. Keep data, runs and dist out of Git history.
To reproduce packaging into a fresh destination:

```sh
uv run python scripts/package_results.py --root runs/study-polar-four --output dist/new-snapshot
```

The packager validates that all four retained runs completed and use seed 0.
It saves archive hashes and a run inventory in `artifact-manifest.json` and never
overwrites ZIPs. Use a new release tag for each snapshot. Automatic training
resume is not implemented. See the [study protocol](study.md).
