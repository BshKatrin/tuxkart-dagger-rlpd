# Study protocol

Train separate policies on Lighthouse, Zen Garden, and Snow Tux Peak. Compare
BC, DAgger, SAC, SAC with symmetric sampling, and RLPD, with five independent
training seeds per method and track. There is one dataset size and one expert.
Evaluate only normal driving, with fixed evaluation seeds 20000–20003.

## Shared task

One kart, one lap, 0.1-second actions, and at most 1,500 actions per episode.
Use the original Team 4 coordinator and parameters at commit
`ddf2cb6573850f890bba26d4e5f82888d9dac82a`.
[Provenance](../../src/kartstudy/_team4/PROVENANCE.md) records compatibility changes.
Reset all expert state once per seeded episode. Advance it once per observation
along the actual trajectory; cache labels so querying them has no side effects.

All policies observe velocity, track geometry, lateral displacement, steering
limit, progress, energy, held powerup, and the nearest 16 item positions/types
with presence masks. Track geometry includes eight upcoming path segments.
The expert receives full native observations; learners receive fixed-size
features. In particular, expert recovery/avoidance memory is not supplied to
feedforward learners: this is imitation of a history-dependent teacher from
current observations, not an assertion that the task is fully observable.

Actions are steering, throttle, brake, drift, nitro, rescue, and fire. Learners
use seven continuous latent controls in [-1, 1]; throttle maps to [0, 1] and
binary controls trigger above zero. This is a continuous relaxation for SAC.
The expert's original commands execute without disabling any control.
Native physical rescue is available to all policies.
Reward is signed progress / 10 − 0.1 per action + 10 on completion.

## Data and learning

Collect 22 noise-free expert laps per track with seeds 1000–1021. Episodes
0–19 are training; 20–21 are held-out validation, excluded from replay and
imitation updates. Collection requires every lap to finish. Metadata includes
expert identity, schema, split, and checksum. Each method uses the same dataset;
online SAC does not sample it. No noise is added to expert actions.

BC fits 2,000 updates. DAgger starts with BC, then aggregates labels on visited
states in 2,000-step rounds, with 500 supervised updates per round and expert
execution probability 0.5 × 0.5^round. Reward learners use 32-step collection
chunks, 1,000 warmup actions, and 100,000 online actions. Actor/critic networks
have two hidden layers of width 128; batch size 256, learning rate 0.0003,
discount 0.99. SAC uses two critics. RLPD uses ten critics, critic layer
normalization, and four critic updates per environment step. SAC with symmetric
sampling isolates the effect of mixing equal online and demonstration samples.
The actor updates once per environment step after warmup.

## Evaluation and reporting

Evaluate deterministic CPU policy snapshots every 10,000 online actions and at
completion, on four fixed seeds per track. BC is evaluated after fitting.
Report completion, percentage of race time off-track, native rescues per lap,
and successful lap time. Retain track length / successful lap time as a secondary
pace diagnostic. Road status is sampled every action and weighted by race time;
the countdown is excluded and the finish interval is clipped. Rescue counts use
native rescue animations, excluding explosions, jumps, and reversing maneuvers.
Expert diagnostics also report inferred harmful contacts / initial harmful rows;
see [measurement limits](../expert-evaluation.md). Do not interpret pace as a gap
to an optimum. Average each metric within a training run first; independent
training runs are the statistical unit. Summaries bootstrap training seeds,
not laps. Use completed runs at matched budgets for final comparisons.

Use schema 2 configurations, datasets, and checkpoints. The viewer and summaries
select this schema, and training validates dataset identity and checksum.
Store source snapshots and immutable datasets with every suite. Use fresh paths
for new runs; automatic resume is not implemented.
