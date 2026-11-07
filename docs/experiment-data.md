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

## SAC symmetric sampling add-on

After restoring `polar-four-seed0-v1`, extract `sac-sym-policy.zip` from the newer
`sac-sym-seed0-v1` release at the repository root. It includes the completed
`sac_sym` seed-0 actor, official evaluations/trajectories, original source snapshot,
configuration comparison, and an updated selection manifest registering all five
methods in the default viewer. It reuses the exact demonstration dataset from
the base release (20 training seeds per track). `sac-sym-training-checkpoint.zip`
is optional and preserves the full original checkpoint. Verify the add-on's
SHA256SUMS. Other policies and extra runs are unchanged.
