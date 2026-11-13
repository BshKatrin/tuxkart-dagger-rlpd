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

## Monte Carlo critic calibration

The separate [monte-carlo-five-disturbances-v1 release](https://github.com/BshKatrin/tuxkart-dagger-rlpd/releases/tag/monte-carlo-five-disturbances-v1)
contains the completed diagnostic for SAC, SAC + SS and RLPD, training seeds 0--2.
It includes 2,700 stochastic continuations: 5 disturbances, paired ordinary and
perturbed states, 9 trained policies, and 30 action-noise seeds per state/policy.
It does not replace the training-policy releases above.

| Archive | Contents |
|---|---|
| `monte-carlo-rollouts.zip` | Original manifests, frozen control tapes, observations/actions/rewards/log densities for all 2,700 rollouts, signed-error summaries, source archives, original recovery bank, and 9 shared-state post-delay recovery trials |
| `monte-carlo-inputs.zip` | 9 exact actor exports and compact evaluation checkpoints with all actor/critic/temperature weights, configs/statuses, training source archives, portable input list, path mapping, and the report matrix PDF/PNG with paired CSV |

The compact checkpoints are derived evaluation artifacts, **not full training
checkpoints**: replay, optimizers and RNG state are omitted. Their provenance
records the original full-checkpoint SHA-256, and loading verifies exact actor
agreement. Original run manifests and source snapshots are preserved unchanged.
No previous release is needed for this diagnostic. Full training checkpoints
remain available in their original releases.

From a checkout containing the release's code:

```sh
gh release download monte-carlo-five-disturbances-v1 --dir dist/mc-download
(cd dist/mc-download && shasum -a 256 -c SHA256SUMS)
unzip -n dist/mc-download/monte-carlo-rollouts.zip
unzip -n dist/mc-download/monte-carlo-inputs.zip
uv sync --locked
uv run python -m kartstudy.critic_artifacts
uv run --with matplotlib --with seaborn python report/scripts/critic_error_recovery.py --output report/generated/critic-error-recovery-report
```

The verifier recalculates every soft return and bias, checks each source/trace
hash, reloads the exported critics, and validates recovery evidence. The path
mapping resolves preserved absolute references relative to the restored checkout.
Historical `analyze.py`/`run_all.py` files retain their original paths; use the
portable verifier and plotting command above on another machine.

To repeat a case in a fresh output directory:

```sh
uv run python -m kartstudy.critic_mc \
  --inputs runs/critic-mc-inputs-v1/policy-inputs.json \
  --reference runs/five-disturbances-v1/evaluation/delay_corner_hacienda--expert.json \
  --output runs/new-mc-hacienda
```

The NPZ files contain numerical evidence, not videos or per-step world poses.
Saved-trace analysis needs no Metal device; exact simulator replay still checks
the frozen state signature. For delayed controls the matrix uses the common
expert-generated post-delay state, rather than each policy's original latency
trajectory. See the [disturbance protocol](disturbance-study.md).

Package a fresh snapshot without overwriting existing exports or ZIPs:

```sh
uv run python scripts/package_results.py --monte-carlo --output dist/new-mc-snapshot
```

## Supplemental 30-seed evaluations

The [reevaluation-thirty-seeds-v1 release](https://github.com/BshKatrin/tuxkart-dagger-rlpd/releases/tag/reevaluation-thirty-seeds-v1)
preserves two completed collections for five learned methods, training seeds
0–2, and one shared expert, with evaluation seeds 20000–20029:

- `reevaluation-metrics.zip`: 6,240 episodes on the original thirteen tracks,
  including configurations, provenance, source snapshot, and per-track metrics.
- `report-trajectories.zip`: 3,840 episodes and full trajectories on the eight
  report tracks, including supplemental held-out Zen Garden.
- `reevaluation-inputs.zip`: exact actor exports at their original repository-relative
  paths and the report's road geometry and track selection.

No earlier release is required to analyze these trajectories. Verify `SHA256SUMS`
and extract all three ZIPs at the repository root. Original manifests preserve
historical absolute paths; resolve those relative to the restored checkout when
repeating evaluation. The inputs contain actor exports, not resumable training
checkpoints. Earlier training and diagnostic releases remain unchanged.

```sh
gh release download reevaluation-thirty-seeds-v1 --dir dist/reevaluation-download
(cd dist/reevaluation-download && shasum -a 256 -c SHA256SUMS)
unzip -n dist/reevaluation-download/reevaluation-metrics.zip
unzip -n dist/reevaluation-download/report-trajectories.zip
unzip -n dist/reevaluation-download/reevaluation-inputs.zip
uv run --with matplotlib --with seaborn python report/scripts/prepare_performance.py
uv run --with matplotlib --with seaborn python report/scripts/performance_scorecards.py \
  report/generated/performance-thirty-seeds/seed-metrics.json \
  --output report/generated/performance-thirty-seeds --capped-time-bars
uv run --with matplotlib --with seaborn python report/scripts/segment_comparison_draft.py --aggregate-training-seeds
```

The [supplemental protocol](protocols/reevaluation-thirty-seeds.md) defines the
statistical units and the report's 150 s failure score. To package a fresh snapshot:

```sh
uv run python scripts/package_results.py --reevaluation --output dist/new-reevaluation-snapshot
```
