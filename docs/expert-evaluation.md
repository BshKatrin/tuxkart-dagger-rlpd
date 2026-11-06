# Expert track diagnostics

```sh
uv run python -m kartstudy.expert_evaluation \
  --output runs/expert-metrics/metrics.json
```

The original Team 4 coordinator drives one lap per seed 0–3 on all 12 training
tracks. The shared five controls execute; drift and explicit rescue stay off. There is no
added action noise. Native simulator seeds can affect powerups and trajectories.
Override `--tracks` or `--seeds` when needed; output paths must be fresh.
JSON records per-seed metrics, arithmetic means, completion, item-row membership,
contact events, source hashes, package versions, and the evaluation protocol.

- **Lap time:** official simulator finish time in seconds; null on failure.
- **Off-track time (%):** percentage of elapsed race time for which the native
  `is_on_road` flag is false. Integrate post-action status using actual simulator
  time increments; exclude the countdown and clip the final interval to official
  finish time. This is a 0.1-second sampled road-status measure, not wheel contact
  or distance from the centerline. A jump above the road need not be off-track.
- **Rescues per lap:** count distinct native rescue animations, including automatic
  and requested rescues. Count animation onset or a restarted timer, not each active
  frame or rescue-button request. Explosion and cannon animations are excluded;
  the expert's reversing recovery maneuvers are not simulator rescues.
- **Average progress speed:** native track length / lap time in m/s; null on failure.
- **Harmful-contact proportion:** inferred banana contacts / initial harmful rows.
  Null if there are no harmful rows. Count every seed equally when averaging.

Rows group banana positions projected onto the nearest 3D centerline segment,
with a maximum 3 m longitudinal span, including across the lap seam. Saved
memberships allow inspection; rows are geometric estimates, not native labels.
Failure records retain the full-track denominator and mark incomplete exposure;
low contact ratios on incomplete laps are not evidence of good avoidance.

PySTK2 exposes neither item availability nor a native pickup counter. Contacts
are inferred from banana-attachment onset/extension and premature bomb removal,
excluding ordinary expiry. Sampling every 0.1 seconds may merge contacts.
Item use can change item types or drop bubblegum; the detector does not measure
those extra contacts or update the initial-row denominator. Tracks with initial
bubblegum are rejected. This is an estimate of banana contacts, not an exact
log of every harmful-item collision. Completion is always reported alongside it.
