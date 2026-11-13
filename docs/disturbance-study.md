# Exploratory recovery from expert failures

This is a post-training case study, separate from official assessments. A spatial
checkpoint means a location reached by a fixed expert prefix, not a training
checkpoint. No learning updates, dataset edits or checkpoint selection occur.

## Matched physical interventions

Reset one headless native simulator with a fixed seed, let the original expert
drive until native progress first crosses `anchor_m`, then override specified
controls for a fixed number of 0.1-second actions. The expert advances once per
observation throughout the prefix, disturbance and response; never reset its
memory at handover. No teleport, fabricated observation, drift or explicit rescue
is introduced. All controllers receive a bit-identical recorded observation,
position, orientation, velocity and native time at handover. This verifies the
observed state, not a complete serialization of every hidden engine variable.

All trials retain the original 1,500-action total episode horizon, including the
prefix/disturbance. Native rescue remains enabled and is recorded separately.
Each trial saves the forced and commanded actions, positions, phases, road status,
speed, progress, rescue telemetry and endpoint. Full expert and learned policies
are evaluated to completion/timeout, rather than stopping after a poor response.

Independent recovery requires 21 consecutive on-road post-handover observations
(spanning two seconds), monotonically advancing at least 10 m, before any native
rescue. Report recovery within 15 s, eventual recovery, lap completion, conditional
successful lap time and rescue count separately. Recovery does not imply finishing
the rest of a held-out track. A native rescue after a valid recovery does not erase
that earlier local recovery, but prevents a rescue-free full-lap claim.

## Qualified locations

The initial search used expert continuations and independent scripted/geometric
recovery attempts. The retained challenges require that the expert times out
without independent recovery or native rescue, while a valid scripted correction
followed by the unchanged expert completes the lap without rescue. Locations and
control variants were frozen before evaluating learned policies.

Both spatial checkpoints use evaluation seed 20000 and the first crossing of
2 m native progress:

- **Lighthouse (training):** expert prefix of 27 actions; full left steering and
  full throttle for 30 actions (3 s), brake/nitro/fire off. A witness brakes with
  straight steering and zero throttle for 15 actions, then returns to the expert.
- **Hacienda (held out in the original thirteen-track protocol):** expert prefix
  of 28 actions; the same left/full-throttle disturbance for 20 actions (2 s).
  A witness brakes with full right steering and zero throttle for 20 actions,
  then returns to the expert.

Each location also has an undisturbed control, a 0.6 s left pulse, and an
opposite-direction severe pulse. JSON controls use the study's normalized [-1,1]
action convention: throttle -1 means zero, throttle +1 means full; Boolean
controls activate strictly above zero. The two scripted witnesses prove
recoverability of their severe-left conditions only; they do not certify every
other disturbance variant.

The completed local case study is under ignored
`runs/disturbance-study-20261110/`: frozen `scenarios.json`, `policy-inputs.json`,
search scripts/ledgers and qualification trials, then `evaluation/` with 130
matched trials: eight expert trials, two witnesses and 120 learned-policy trials
(five methods × three Mac training seeds × eight conditions). Seeds 0/1/2 occur
once per method; overlapping released/local policies are not counted twice.
The evaluation manifest preserves policy hashes, source archive, versions,
platform, scenario definition and per-trial hashes. A discontinued witness search
and the initial empty serialization-failure ledger are retained as search history;
only completed evaluation-manifest entries enter summaries.

## Reproduction and interpretation

```sh
uv run python -m kartstudy.disturbance \
  --scenarios runs/disturbance-study-20261110/scenarios.json \
  --input PATH_TO_RUN [ANOTHER_RUN ...] --output runs/new-disturbance-evaluation
uv run --with matplotlib python scripts/analyze_disturbances.py \
  runs/new-disturbance-evaluation
```

Collection requires a fresh output directory. Analysis verifies cached hashes and
produces per-condition CSV/JSON and recovery PNG/PDF maps. Maps illustrate the
predeclared training seed 0 and first 15 s of response with a fixed local view;
summary tables retain all three training seeds and full-episode outcomes.

These two challenges expose an early-race expert weakness. Their failures are
consistent with its hardcoded straight-driving branch at progress <=2 m and
stuck-detection gate above 10 m; late expert actions are full-throttle/zero-steer
while progress remains zero. This is not proof of broad OOD robustness, nor of
where a learned policy's training support ends. The expert-only selection avoids
engineering a BC failure; observed BC recovery must remain in the result. One
location per split and one evaluation seed provide case evidence, not a
population-level method ranking. The held-out outcomes are exploratory and do not
tune the original study or choose its retained checkpoints.

## Mid-race follow-up validation

Use later native-progress anchors to distinguish early-race controller guards
from other recovery failures. An apparent timeout alone does not establish a
persistent inability to recover: repeat its exact prefix/intervention with a
longer diagnostic horizon, verifying identical handover state. Preserve the
original-budget outcome separately; extended trials are not matched-budget study
replicates. Inspect credited progress resets and actual world motion before
calling an outcome a stationary block, a spatial loop, or a successful local
recovery. Passing a local recovery criterion can precede a much later finish.

Record automatic rescues before the intervention, during it, active at handover,
and after handover separately. A state reached after an earlier automatic rescue
can be compared through an identical simulated history, but is not a clean
rescue-free perturbation. Do not describe earlier rescues as evidence of failure
after handover; do not count an active rescue animation as controller recovery.

## Frozen action-selection and control interventions

To test whether deterministic evaluation contributes to a particular failure,
compare the same frozen CPU actor's `act(obs)` (`tanh(mean)`) with its original
`forward(obs).value` (a tanh-transformed Gaussian sample). Use the same simulation
seed, expert prefix, disturbance and episode horizon. Give sampled rollouts
explicit action-noise seeds, pairing those seeds between methods, and preserve
all handover signatures. These repetitions measure action-selection variability
of one policy, not additional independent training runs. Keep official evaluation
results unchanged and report any sampled results as separate diagnostics.

Compare actor outputs on identical observations as well as their own rollouts;
different trajectories alone cannot isolate decision differences. For Boolean
controls triggered above zero, compute decision probabilities from the latent
Gaussian rather than treating the deterministic control as an average physical
action. A deterministic failure that persists under sampling is not explained
solely by omitting exploration noise.

A further local intervention can override one control channel while the frozen
actor selects its other outputs on the newly visited states. Record the exact
override rule and full closed-loop continuation. For example, disable brake,
borrow steering from another frozen actor, or combine the two. Successful
interventions establish sufficiency for that state and controller, not a general
algorithm ranking or the cause of a difference in training. No such hybrid should
be reported as the performance of the original trained policy.

## Recovery across training checkpoints

Apply the same frozen scenario bank to genuine intermediate actors:

```sh
uv run python scripts/disturbance_evolution.py collect \
  --scenarios runs/disturbance-evolution-v1/scenarios.json \
  --input PATH_TO_RUN [ANOTHER_RUN ...] --output runs/new-snapshot-collection
uv run --with matplotlib python scripts/disturbance_evolution.py render \
  --root runs/new-snapshot-collection --output runs/new-snapshot-figures
```

The scenario list includes `reference_result` and `reference_result_sha256` fields
pointing to each selected original expert result. Collection verifies those hashes,
replays the original expert history, and requires every snapshot's handover to
match that original case exactly. Inputs use the existing snapshot discovery and
provenance checks. Source archives, actor hashes, full episode traces and results
are preserved in fresh output directories. Rendering accepts multiple completed
collections (for example, one per training seed), rejects duplicate actor trials,
and verifies that repeated fixed expert references agree.

Use the available seed-1/2 histories at 0, 40k, ..., 400k actions; do not infer
seed-0 history from final actors. BC remains a fixed fitted reference at all
stages; DAgger's zero-action actor is already BC-initialized. Other zero-action
actors precede online training. For each case, a recovery map uses four categories:
no timely recovery/timeout, no timely recovery/finish, timely recovery/timeout,
and timely recovery/finish. Timely recovery means the same rescue-independent
15-second criterion. Full-lap completion can include later native rescue; CSV/JSON
retain rescue counts and conditional lap times rather than hiding that distinction.

Each GIF advances training checkpoints and displays the complete first-15-second
response trajectory for every method, with fixed map bounds. Driving time does
not advance within that GIF. Expert and BC panels remain explicitly fixed.
Individual PNG frames support later static figure selection. These are case
studies of learning and regression on preselected disturbances; inspecting
intermediate held-out behavior must not change official checkpoint selection.

## Five disturbance families and paired controls

The follow-up bank `runs/five-disturbances-v1/scenarios.json` freezes one physical
case per requested family before evaluating final actors. Each has a separate
undisturbed continuation from the same expert-prefix anchor. Every method uses
its Mac training seeds 0, 1 and 2 exactly once; the expert is one fixed controller,
not three independent runs. The bank contains 162 trials: ten conditions ×
(expert + fifteen learned policies), plus two scripted witnesses.

| Family | Track / split | Native-progress anchor | Physical intervention |
| --- | --- | --- | --- |
| Full left + throttle | Lighthouse / training | 2 m | Original 3 s full-left/full-throttle pulse |
| Stopped against geometry | STK Enterprise / training | 850 m | 1 s full-left/full-throttle; handover at 0.059 m/s |
| Facing backward | Cornfield Crossing / training | 400 m | 1 s straight braking, 3.5 s full-left at 50% throttle, 0.5 s straight braking; heading error 177.7° near the road edge |
| Alternate branch | Black Forest / training | 560 m | Fixed 1.4 s geometric-guide control tape enters native branch node 1094 |
| Delayed corner controls | Hacienda / held out | 375 m | 0.6 s actuator latency for 2 s, then immediate normal control |

The Enterprise blocking probe moves only 0.16 m under 2 s straight full throttle;
straight reversing moves 3.45 m and permits an expert finish. The expert itself
also escapes the selected barrier state. New cases need not defeat the expert:
recoverability and actual policy responses are the questions. Selection uses
expert/independent qualification rather than learned-policy failures.

Black Forest's alternate branch is a legal native route, not intrinsically a
wrong route. The intervention differs from the expert's unperturbed route at this
junction. Inspect native node sequences and complete continuations: two seconds
of stable driving after entry do not establish successful branch traversal or
lap completion. This case does not by itself prove absence from demonstrations.

Latency is an actuator queue, not frozen observations or a repeated single action.
Initialize it with the last six actually executed expert-prefix controls. At each
of twenty delayed steps, execute the oldest queued action and append the
controller's current requested action. Then discard pending controls and execute
the current request immediately. Save `requested_action` separately from executed
`action`, together with `delay_active` and the control-restoration signature.
Controllers share the exact state when delay begins; their states when it ends
can differ because their intervening requested actions differ.

For latency, start the two-second recovery window and 15-second recovery deadline
at control restoration; count any rescue during the delayed interval against
independent recovery. Other cases retain handover as their recovery-clock origin.
The total 1,500-action episode budget remains unchanged. All cases additionally
record native road node and kart-to-road heading angle without changing physics.

Report each training seed beside its paired undisturbed result. In particular,
an existing baseline failure cannot be attributed to the disturbance, and lap
completion after an automatic rescue must not be called unaided recovery. These
five selected cases support mechanism-oriented comparisons, not a robustness
ranking over tracks, simulator seeds or perturbation magnitudes.

Interpret control telemetry using native semantics: the installed simulator
prioritizes any nonzero acceleration over the braking/reverse branch. A positive
brake output together with positive throttle therefore does not establish that
physical braking occurred. The study decoder maps throttle continuously as
`(action[1] + 1) / 2`; preserve this mapping in comparisons. An explicit
brake-priority diagnostic (force throttle to zero whenever brake is requested)
is a controller intervention, not the original trained policy's performance.
Keep its results and provenance separate; do not silently fix decoding for only
one method or retrospectively replace completed evaluations.

## Frozen-critic Monte Carlo signed error

Use `kartstudy.critic_mc` to compare the mean online critic prediction with an
independent soft-return estimate. This diagnostic uses final SAC, SAC + symmetric
sampling and RLPD checkpoints, with exactly matched portable actor weights and
training seeds 0--2. Do not substitute a different run with the same seed.

For Lighthouse, replay the original expert prefix to 2 m, then either continue
the expert for 30 actions or apply the frozen 30-action full-left/full-throttle
pulse. The two measurement states have the same native time and prefix length;
the perturbed state must match the original recovery experiment exactly. This
ordinary control is later than the prefix-only control in the recovery bank.

For each state and policy, fix the first action to `tanh(mean)` and evaluate the
online critic ensemble mean at that exact normalized action. Use 30 independently
seeded stochastic policy continuations afterward, pairing action-noise seeds
between policies and conditions. CPU inference, physical action decoding, reward,
observations and native rescue behavior remain unchanged. No recovery, completion
or lap-time comparison is computed in this diagnostic.

The Monte Carlo return is the discounted reward sum plus future entropy bonuses
`-alpha * log_prob`, using the saved temperature and discount. The fixed first
action contributes no entropy term, matching the critic's Bellman convention.
Signed error is `b = mean(Q_i) - mean(MC soft return)`. Positive values indicate
optimism and negative values indicate pessimism at the selected state/action;
this is not a comparison with the optimal recoverable value.

Continue until a genuine lap finish or 2,500 response actions. An artificial
cutoff does not become a terminal-zero training target. For unfinished rollouts,
check that the discounted contribution after action 1,500 is below 0.001 return
units; the remaining discount at 2,500 actions is about 1.2e-11 when gamma=0.99.
Never bootstrap a Monte Carlo reference from the critic being evaluated.

Report each training seed's signed error and Monte Carlo uncertainty separately,
then average signed errors equally across the 3 trained policies. Monte Carlo
repetitions are not independent training runs. Frozen signatures, complete control
tapes, actor/critic/source hashes, per-action observations, actions, rewards and
log probabilities are retained under a fresh ignored output directory.

```sh
uv run python -m kartstudy.critic_mc \
  --inputs runs/disturbance-study-20261110/policy-inputs.json \
  --reference runs/five-disturbances-v1/evaluation/left_throttle_lighthouse--expert.json \
  --output runs/new-critic-mc
```

The same signed-error protocol also applies to the 4 remaining frozen cases:
Enterprise barrier, Cornfield wrong direction, Black Forest alternate branch,
and Hacienda temporary latency. Use the original expert result for `--reference`.
For pulse/tape interventions, measurement is immediately after the intervention;
its undisturbed control follows the expert for the same number of actions. Keep
one native simulator per worker process and keep all policy inference on CPU.

For temporary latency, measure at normal-control restoration, not delay onset.
The original expert supplies requested controls during the delayed interval;
the fixed queue is initialized with its actual prefix actions. The perturbed
signature must equal that original expert trial's `delay_release`. The control
continues the undelayed expert for the same duration. Discard queued controls
before Monte Carlo continuation so the transition dynamics again match training.
This is one shared expert-generated delayed state for every critic; it is not
an evaluation of each actor's different state after experiencing latency itself.
Store and replay the full executed control tape and verify the same physical
state before every continuation. The continuation budget is 2,500 actions beyond
the measurement point, including the extra delayed interval in prefix accounting.

For the signed-error/recovery figure, pair each training seed's perturbed-state
`b` with deterministic recovery within 15 s at that exact state. This compares
stochastic-value calibration with deterministic behavior. Reuse pulse/tape
recovery labels only after verifying actor hashes and handover signatures.
Reevaluate latency from the shared expert-generated post-delay state; the original
policy-specific latency outcomes cannot supply these labels. Preserve the new
traces separately. `report/scripts/critic_error_recovery.py` produces the figure
and its paired data. The matrix uses disturbance rows, method columns, and one
bar per training seed at a shared bias scale; no policy averages are plotted.
