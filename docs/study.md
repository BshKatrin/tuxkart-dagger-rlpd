# Study protocol — shared policies, schema 3

Compare BC, DAgger, online SAC, and RLPD using one retained policy per method
(training seed 0, selected consistently without ranking evaluation performance).
Additional training-seed checkpoints are excluded from this comparison.
The original five-seed protocol is preserved in the suite source archive. Each run learns one feedforward policy shared across all four training
tracks. Track names and difficulty are not policy inputs. Difficulty is used
only to group reported results. This protocol replaces the
[three-track schema 2 protocol](protocols/study-schema2.md); existing artifacts
and source snapshots remain unchanged. The [12-track revision](protocols/study-twelve-tracks.md)
was superseded to reduce compute; its runs are excluded from this study.

## Track split

| Difficulty | Training | Held out from learner training |
|---|---|---|
| Simple | Lighthouse, Cornfield Crossing | Antediluvian Abyss, Candela City, Cocoa Temple, Hacienda, Volcan Island |
| Medium | Black Forest | Old Mine, Northern Resort |
| Hard | STK Enterprise | Ravenbridge Mansion, Fort Magma |

The [track mapping](../src/kartstudy/tracks.py) fixes the simulator IDs, including
`mines` (Old Mine) and `snowmountain` (Northern Resort). The study uses four
training and nine held-out tracks. The eight other standard tracks are unused
in this revision. Held-out tracks have been inspected when
qualifying the expert; they are held out from learner training, rather than
untouched throughout study design. They supply no demonstrations, replay,
DAgger labels, fitting data, or model-selection feedback.

## Shared task

One kart, one lap, 0.1-second actions, and at most 1,500 actions per episode.
Use the original Team 4 coordinator and parameters at commit
`ddf2cb6573850f890bba26d4e5f82888d9dac82a`.
[Provenance](../src/kartstudy/_team4/PROVENANCE.md) records compatibility changes.
Reset expert state once per episode; advance it once per observation along the
actual trajectory, caching labels so repeated queries have no side effects.

All learners receive the same 84-value observation: the existing 65 road/motion
features (velocity, road-center vector, lateral distance, steering limit,
progress, and eight upcoming segment starts/ends/widths), energy, one-hot held
powerup type (11 values), held-powerup count, and six hazard values.
Keep only bananas and bubblegum satisfying the expert's radar: |x| ≤ 2 m,
0 ≤ z ≤ 17 m, and |center-path-distance + x| ≤ first-segment-width / 2.
Sort by forward distance z, select two, and encode each as
`(sqrt(x²+z²)/sqrt(293), atan2(x,z)/(pi/2), present)`.
An absent slot is `(0, 0, 0)`; no hazard-type encoding is needed.
Keep the existing scaling and clipping for other features.

Actions are steer, throttle, brake, nitro, and fire. All five network outputs
are continuous latent controls in [-1, 1]; throttle maps to [0, 1], while
brake/nitro/fire trigger strictly above zero. This continuous relaxation is
shared by SAC and RLPD and is not a discrete-action SAC formulation.
Drift and explicit rescue are fixed off (the original expert also leaves them
off); native automatic rescue remains available. No recovery features, memory,
or shared expert overrides are added. The expert retains its own recovery and
avoidance history, which a current-observation MLP cannot reproduce exactly.
Reward remains signed progress / 10 − 0.1 per action + 10 on completion.

## Data and learning

Qualify the expert on every training track before collection. Collect 22
noise-free expert laps per training track, using seeds 1000–1021. Within each
track, the first 20 laps train and the last two validate: 80 training and eight
validation laps in one pooled dataset. Validation episode IDs are explicitly
excluded from replay and imitation updates. Executed actions equal cached
expert labels. Metadata records tracks, split, expert, schema and checksum.
The fresh four-track dataset is an exact subset of the original noise-free collection, with original
episode IDs and parent-dataset checksum retained; no checkpoints are reused.
Every lap must complete; failures stop collection rather than being discarded.

Every online run receives exactly 400,000 actions: 100,000 on each training
track. At an episode boundary, select the training track with the fewest actions
so far (ties follow the fixed track order). End a collection episode at its
track's exact quota; this is a sampling boundary, not a terminal Bellman target.
Periodic evaluation also interrupts collection and resets before training resumes.
Online SAC uses no demonstration samples. RLPD samples equal numbers of online
and pooled training demonstrations. Sampling within each source is uniform over
transitions. The same pooled demonstrations initialize BC and DAgger.

The MLPs retain two hidden layers of width 128, batch size 256, learning rate
0.0003 and discount 0.99. To preserve the previous fitting and collection budget
per track, multiply the former per-track budgets by four: BC fits 8,000 updates;
DAgger starts with BC, then uses 8,000-action rounds with 2,000 supervised
updates per round and expert execution probability `0.5 × 0.5^round`.
Reward learners collect 32-action chunks and use 4,000 warmup actions.
SAC uses two critics and one critic update per action. RLPD uses ten critics,
critic layer normalization, and four critic updates per action. Both update the
actor once per action after warmup. This suite omits the separate SAC symmetric
sampling ablation, so it does not isolate symmetric sampling from RLPD's other
components.

## Evaluation and reporting

Use deterministic CPU snapshots, four fixed evaluation seeds 20000–20003 per
track. Evaluate training tracks every 40,000 online actions. Evaluate all 13
study tracks at the fixed final budget; BC is evaluated after fitting. Held-out
results do not select checkpoints or tune the methods. Final trajectories are
saved separately per track. Training uses Metal; simulation and inference use
CPU, with one native simulator per worker process.

Report completion, race-time-weighted off-track percentage, native rescues per
lap, and successful lap time. Keep track length / successful lap time as a
secondary pace diagnostic. Always report completion beside conditional lap time;
see [measurement limits](expert-evaluation.md). Average evaluation laps within
each method first. With one retained training run per method, do not report
training-seed confidence intervals or claim robustness across initializations.
Repeated evaluation laps measure evaluation variability, not training variability.
Report each track, each split/difficulty group, and each split overall. Groups
pool their fixed evaluation laps within each run. A group with no successful
laps has no conditional lap time. Summaries can contain partial runs; use only
completed runs at matched budgets for final comparisons.

Configurations, datasets and checkpoints use schema 3. A suite validates the
complete dataset before queuing jobs, saves exact source and protocol snapshots,
and records per-track online counters in statuses and checkpoints. Use fresh
artifact paths. Training resume is not implemented.

## Additional 20-demo symmetric-sampling ablation

A separate `sac_sym` run uses training seed 0 and the same 20 demonstration
seeds (1000–1019) per training track. Validation remains 1020–1021, excluded from
replay. It uses the same four tracks and 400,000-action budget, two critics,
no critic layer normalization, one critic update and one actor update per action
after warmup, and 50/50 online/demonstration sampling. Other SAC settings and
fixed evaluation seeds are unchanged. This matches the demonstration budget of the retained SAC/RLPD methods,
so SAC versus SAC symmetric sampling isolates the sampling change.
Its artifacts and source snapshot live separately from the stopped main suite;
its result is not included in the four-policy release.

## Five-method comparison and reward-method replication

The separate five-method comparison adds the completed 20-demo `sac_sym`
ablation to the retained BC, DAgger, SAC and RLPD policies. It preserves the
four-policy release and its protocol above. SAC versus `sac_sym` isolates
50/50 sampling; `sac_sym` versus RLPD also changes critic count, critic layer
normalization and update-to-data ratio. These are fixed-budget comparisons,
not equal-compute comparisons.

Generate the comparison from explicit completed run directories; choose a
fresh output directory each time:

```sh
uv run --locked python scripts/compare_methods.py \
  --runs runs/study-polar-four/shared_bc_s0 \
         runs/study-polar-four/shared_dagger_s0 \
         runs/study-polar-four/shared_sac_s0 \
         runs/ablation-sac-sym-demo20/sac_sym_s0 \
         runs/study-polar-four/shared_rlpd_s0 \
  --output runs/analysis/five-method-seed0-new
```

The command verifies schema, budgets, complete assessment grids, dataset
checksum and dependency versions. It writes final results by track,
split/difficulty and split; recorded training curves; compute counters; JSON,
CSV and Markdown tables; PNG/SVG figures; input provenance and output checksums.
It refuses stopped runs, duplicate method/seed pairs, edited inputs and existing
outputs. `--no-figures` retains the tables and provenance.

For replication, retain seed 0 unchanged and train SAC, `sac_sym` and RLPD at
seeds 1 and 2 in a separate source snapshot and output directory. Preserve the
same demonstration checksum, four training tracks, 100,000 actions per track,
400,000-action total, 40,000-action assessments and 13-track final evaluation.
BC and DAgger remain at one training seed. Add a replication run to `--runs`
only after its final assessment and completed status are available; every
figure and table must retain the resulting unequal method seed counts.

Pool evaluation laps within each training run, using
`sum(off_track_seconds) / sum(measurement_seconds)`, then average independent
training runs equally. Bootstrap training runs, never evaluation laps. A single
training seed has no training-seed interval; intervals from three seeds remain
exploratory. Conditional lap time excludes failed laps and has no estimate for
a seed with zero completions; retain completion denominators and the number of
seeds contributing to each metric. Do not substitute launcher progress summaries
for these final statistics. BC's fitted assessment is a reference point, not a
historical online learning curve. Genuine evolution replays use recorded actor
snapshots from new runs; final actors cannot reconstruct past training states.
