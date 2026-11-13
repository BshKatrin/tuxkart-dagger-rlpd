# Local time-loss maps

The report compares all five retained seed-0 policies against the original expert.
Each policy has a 2 × 4 map: the four Training tracks in
[tracks and policies](tracks-and-policies.md) on top, followed by its four Held out
tracks. This report selection does not change the completed suite's protocol,
training split, source archive, checkpoints, or original evaluation files.

## Inputs and reproduction

```sh
uv run python report/scripts/track_time_loss.py collect
uv run --with matplotlib --with seaborn python report/scripts/track_time_loss.py plot
uv run python report/scripts/track_time_loss.py summary
uv run python report/scripts/track_time_loss.py local-summary
```

Collection reads retained completed checkpoints and original final trajectories.
It adds expert references for all eight tracks and missing learner evaluations
for Zen Garden, using deterministic CPU actions, seeds 20000–20003, the existing
0.1-second interval and 1,500-action horizon. No training or checkpoint selection
uses these results. The native simulator is reused sequentially in one process;
avoid resetting an unstepped warmup race solely to inspect geometry. Geometry is
captured after an evaluation reset, when the expert is first queried.

Outputs stay under ignored `report/generated/track-time-loss/`. `sources.json`
identifies original versus supplemental episodes and hashes input trajectories;
`analysis.json` records timing gates, local means, sample standard deviations,
matched-pair counts and completion. Use a fresh output directory if inputs change.
The script header documents its options. Upload the five vector figure PDFs to
Samskrift and embed those versions; PNG exports are previews. Report sources
remain on Samskrift.

## Measurement

Use first forward crossings of shared native-progress gates approximately every
20 m. Interpolate between consecutive 0.1-second observations; exclude countdown
time and use official finish time at the last gate. Stopping and reversing delays
remain inside sector times. For each matched seed, subtract expert traversal time
from policy traversal time. Normalize by sector length to seconds per 20 m.

Alternative routes are timed as a whole between common split/merge junctions.
Overlapping branch regions are merged. Colors are placed on the native
first-successor reference centerline; other routes appear as a thin background.
This avoids pretending that different branches have identical intermediate
timing gates, but sacrifices spatial detail inside branch regions. Native
progress and 3D road geometry prevent top-down overpasses from being confused.

Show the mean only where all four matched pairs complete the sector without a
sampled native rescue during its traversal. Other sectors are gray, including
unvisited portions after failures. This common-coverage rule avoids averaging
only the surviving laps. Completion remains a separate per-panel count.

All maps share the same symmetric color scale, saturated at ±0.5 s per 20 m;
uncapped values remain in `analysis.json`. Green is faster, red slower. Dashed
segments indicate that the mean ± one sample standard deviation includes zero;
this describes evaluation variability, not statistical significance or variation
across training initializations. Maps retain shape but are individually scaled.

For complete rescue-free laps, sector times must sum to official finish time and
local differences must sum to the paired lap-time gap before length normalization.
Existing evaluation tests also cover reversing delays, rescue censoring,
unfinished sectors, and junction-based grouping.

## Aggregate comparison

`aggregate-metrics.json` reports all tracks, training tracks and held-out tracks
separately, with the expert as a reference. Completion counts retain all attempts.
Successful lap time averages only completed laps and therefore describes different
subsets when completion differs; it must not determine the ranking by itself.

The report's lap-performance and driving-behavior tables additionally include
completed training seeds 0–2 at matched final budgets. Each table cell gives the
equally weighted mean of the 3 run-level summaries, with the observed minimum
and maximum underneath. These ranges describe the observed training runs, not
confidence intervals. Training and held-out tracks remain separate, and the
expert is one shared reference evaluation. The retained seed-0 figures and maps
continue to describe evaluation variability within that controller; their SDs
are not variability across training runs. The completed suite's protocol and
source snapshots remain unchanged.

The updated tables and training-seed capped-time bars use 30 evaluation seeds
(20000–20029) on the same eight report tracks, giving 120 attempts per split and
training run. Regenerate their inputs with `report/scripts/prepare_performance.py`
from the separate report trajectory collection defined in the
[supplemental protocol](protocols/reevaluation-thirty-seeds.md). The bar error bars
are ±1 sample SD across the 30 evaluation-seed means, each averaging four tracks.
They describe evaluation variability within a fixed training run. These report
summaries retain the 150 s failure score used by the previous report tables.
The original seed-0 spatial-map inputs remain unchanged. The Enterprise RLPD
overview now averages the three training runs and 30 evaluation seeds, as
described below; the other spatial figures retain their original inputs.

The original `aggregate-metrics.json` budget-capped completion time uses official
finish time on success and the common
race-clock timeout on failure. The script verifies that every failed attempt is
a timeout at the same horizon (148.075 s in these artifacts), rather than choosing
a penalty to favor a policy. Time is averaged over the fixed tracks within each
evaluation seed, then over the four seeds. The reported sample SD describes those
four seed-level averages; it is not variability across training initializations.

Off-road percentage is total off-road seconds divided by total measured race
seconds, including failed attempts and excluding countdown. Rescues are averaged
per attempted lap, including failures. Low rescue counts alone do not establish
successful driving: a kart can remain stuck without a native rescue.

Use completion first and capped time as a combined budget-dependent pace check.
Keep training and held-out results separate before interpreting the overall mean.
A global signed local time gap adds little: on complete valid laps, its sum is
the expert-relative lap-time gap. Averaging only available colored sectors would
also compare different coverage across policies. Keep local gaps for spatial
diagnosis rather than adding a second global ranking score.

## Saved local summaries

`local-summary` reads the same saved inputs and writes `local-metrics.json`.
It reports four descriptive quantities for all tracks, training and held-out
tracks, with the same equal-track and evaluation-seed aggregation as capped time:

- Signed local time gap: `20 * sum(policy time - expert time) / sum(segment length)`,
  in seconds per 20 m. Gains and losses cancel.
- Number of compared segments: count valid policy–expert pairs for each evaluation,
  then average counts across tracks and evaluation seeds. A segment may contribute
  even if only one seed provides a valid pair. The mean total is 51.75 comparison
  segments per track in each of these three groups.
- Spatial mean speed, in m/s, using the logged 3D velocity magnitude.
- Spatial mean absolute lateral position, as a percentage of nominal road
  half-width. Zero is the centerline; 100% is a nominal edge. Lower is not
  necessarily better because valid racing lines can use the road edges.

For each track and seed, all four quantities share a mask: both policy and expert
complete the segment without native rescue and provide finite profile samples.
Failed laps still contribute earlier completed segments. Missing intervals are
not zero-filled. The existing maps retain their conservative four-valid-pair
display rule; these summaries also use available observations from gray map regions.

The report figure is generated with:

```sh
uv run --with matplotlib --with seaborn python report/scripts/local_performance.py
```

It shows signed local time gap, absolute lateral position, and **Relative
trajectory length (% of expert)**. Segment counts and speed remain in the raw
local-summary export but are not report metrics. For each completed lap, the
relative length is `100 * traveled distance / mean completed expert distance`
on that track. Distance is the 3D polyline through recorded positions, trimmed
to race start and official finish. Intervals touching native rescue activity
or a rescue-count change are excluded; skipped road is not reconstructed.
This is a sampled distance estimate, not native road-segment count or progress.
Only completed laps contribute to the ratio. Available completed tracks are
averaged equally within each evaluation seed, then mean and sample SD are
computed across seeds. Contributing track subsets can differ; interpret the
ratio alongside completion. Time gap and lateral position continue to use
available matched segments, including portions of failed laps. Figure outputs,
individual completed distances, reference means and per-seed contributing track
counts are saved under `report/generated/aggregate-performance/`.

Profiles use first forward crossings at equally spaced midpoints no more than
2 m apart inside each segment. Samples are averaged by native progress distance,
not residence time. Across route branches these are region summaries, not claims
that the karts occupied identical world positions. Different routes share timing
gates; lateral position is measured relative to the locally traveled road.

Lateral position is recomputed from saved world positions and native 3D centerline
segments, restricting candidates to nearby native progress. Its sign uses the
road tangent and global up, so kart heading and jump height do not create false
sideways displacement. Width normalization uses the native full road width.
Road-edge references are approximate on banked roads; the native road-status
flag remains the official off-road measurement. Ordinary off-road offsets are
retained. This replaces the kart-relative signed 3D-distance observation for
the lateral summary only; observations and trained policies are unchanged.

For every field, compute each track's summary within a seed, average tracks
equally, then report mean and sample SD over the four evaluation-seed aggregates.
Coverage and conditional support differ across policies, so this table explains
local behavior rather than replacing the completion/capped-time comparison.

## Static multi-metric overview

The report's RLPD/Enterprise overview uses the same three completed training
runs and 30 evaluation seeds as the updated tables:

```sh
uv run --with matplotlib --with seaborn python report/scripts/segment_comparison_draft.py --aggregate-training-seeds
```

This reads `performance-thirty-seeds/sources-s0.json` through `sources-s2.json`
under `report/generated/` and writes
`report/generated/enterprise-rlpd-thirty-seeds/`. Pair each RLPD trajectory with
the expert trajectory using the same evaluation seed. Average each region's
paired differences within each training run, then average the three run means
equally. The expert's 30 reference trajectories are shared, rather than treated
as independent training replicates. All 61 regions have all 90 paired traversals.
The overview uses solid colors for the regional means. The JSON retains
training-run SDs, per-run means, evaluation SDs, pair counts and source hashes.

The original single-training-run overview can still be regenerated separately:

```sh
uv run --with matplotlib --with seaborn python report/scripts/segment_comparison_draft.py
```

This reuses the saved RLPD/expert STK Enterprise traces and writes the paired
segment data plus `report/generated/segment-comparison-draft/enterprise-rlpd-overview.pdf`
for the report and a PNG preview.
The report figure places two full-track maps side by side, with identical
geometry and scale: signed time gap and difference in absolute normalized
lateral position. Each value is RLPD minus expert. Lateral differences use
percentage points of road half-width and do not encode driving quality.

The caption reports segment coverage. The overview uses solid colored roads
without marker dots or region labels; the speed panel and variability dashes
are omitted. Clipped color scales retain explicit end limits. The figure is a
static overview and requires no selection, hover or zoom to read.

## Progression of trajectories

For chronological maps of saved intermediate policies, use the reusable
[trajectory evolution command](visualization.md#trajectory-evolution-across-training).
It performs separate exploratory evaluations and exports a GIF, individual frames
and compact PNG/PDF strips. On unfamiliar tracks, examining intermediate policies
is post-training exploration; these outcomes must not select checkpoints or tune
methods for the original final held-out comparison. Official final trajectories
and the completed study protocol remain unchanged.
