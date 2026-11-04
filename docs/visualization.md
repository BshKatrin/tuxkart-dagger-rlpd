# Watch the trained policies

Open the native SuperTuxKart viewer:

```sh
uv run python -m kartstudy.playback
```

Select a policy, training seed, experiment, track, and normal or disturbed driving
from the controls above the game. All completed training seeds are available;
unsupported method/experiment combinations are filtered out. Switching away
from an unavailable method selects DAgger for that experiment.

The image comes from the actual 3D simulator. The strip below it shows race
time, speed, progress, and the executed steering, throttle, and brake commands.
Pause, advance one action, restart, or change playback speed. Keyboard shortcuts:
**Space** to pause, **Right arrow** to step, **R** to restart, **Esc** to close.
The renderer can briefly show the game's loading screen at the start of a lap.

Each policy drives alone. Simulation keeps the original 0.1-second action
interval; playback speed changes how quickly we watch it, not the physics.
The optional disturbance uses the same seeded intervention as evaluation.
The expert and learned policies otherwise act deterministically: demonstration
collection noise is off. The expert's conservative variant is available under
the conservative-expert experiment.

To start at a specific policy or evaluation seed:

```sh
uv run python -m kartstudy.playback --policy sac_sym --train-seed 2
uv run python -m kartstudy.playback --policy rlpd --track snowtuxpeak --perturb
uv run python -m kartstudy.playback --policy dagger --experiment small_data --train-seed 1
```

## Record a lap

```sh
uv run python -m kartstudy.playback --policy dagger --train-seed 0 \
  --record runs/visualizations/dagger.mp4
```

Recording runs a fresh episode through completion or timeout, with a sidecar
JSON recording its configuration and outcome. Existing recordings are never
overwritten. Add `--limit-steps 300` for a labeled partial preview. For expert
footage use `--policy expert`; BC and online SAC are `bc` and `sac`.

Recordings sample every policy action at 10 frames per second and preserve the
startup countdown. A two-second hold is added at the end. Graphics may introduce
tiny floating-point differences from headless execution, so original evaluation
logs remain the source for quantitative results.
