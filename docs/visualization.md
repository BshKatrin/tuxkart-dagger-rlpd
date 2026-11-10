# Policy playback

Restore the base release first: extract `selected-policies.zip` and `study-data.zip`
at the repository root. Extract the newer `sac-sym-policy.zip` add-on afterward
to register the fifth learned policy, then run `uv sync --locked`.

```sh
uv run python -m kartstudy.playback
uv run python -m kartstudy.playback --policy rlpd --track stk_enterprise
uv run python -m kartstudy.playback --policy expert --track lighthouse
```

The default is DAgger. Select Policy and Track in the Tk window. The five learned
policies (including the SAC symmetric sampling add-on) all retain **training seed 0**, alongside the original expert reference.
[Tracks and policies](tracks-and-policies.md) lists every CLI name, simulator track
ID, difficulty group and training/evaluation role. The viewer offers all installed
race tracks, including tracks outside the study, with every available policy.

- **Pause / Play** (Space): pause or resume simulation.
- **Step →** (Right arrow): advance one 0.1-second action and pause.
- **Restart** (R): reset the selected episode with the same policy and evaluation seed.
- **Reset simulator**: terminate this viewer's renderer and create a fresh native
  simulator process, retaining the policy, track and evaluation seed.
- **Playback speed**: 0.25×, 0.5×, 1×, 2× or 4× wall-time pacing.
- **Escape / close window**: stop the viewer and its renderer.

Track loading happens in the renderer. Tk stays responsive while requests run;
selectors are disabled until loading completes. A request that exceeds 30 seconds
for loading or five seconds for a step stops the renderer and displays an error.
Click Reset simulator after an error. It does not affect datasets, training or
other simulator processes. Tk and SDL/OpenGL stay in separate processes on macOS.

Playback uses deterministic CPU inference at the original 0.1-second action
interval. Speed changes wall time only. `--eval-seed` defaults to 20000;
`--root` defaults to `runs/study-polar-four`. `--width` and `--height` control
render size (even dimensions, minimum 640 × 360). `--train-seed` is an advanced
checkpoint selector; the distributed selection contains only seed 0.

```sh
uv run python -m kartstudy.playback --policy rlpd --track stk_enterprise \
  --record runs/visualizations/rlpd-enterprise.mp4
```

`--record` saves an MP4 plus JSON sidecar instead of opening Tk. Existing outputs
are refused. `--limit-steps 100` makes a labeled partial preview. Fresh rendering
may introduce tiny floating-point differences; official evaluation logs remain
authoritative and are never replaced by video metadata.

## Trajectory evolution across training

Animate **training action count**, with a complete deterministic lap in each frame.
The map, scale, evaluation seed and 1,500-action horizon stay fixed across all
inputs. The dashed expert trajectory is a spatial reference; native rescue
animations break the driven line, crosses mark rescues, open circles mark timeouts,
and squares mark finishes. Completion and conditional lap time accompany each map.

```sh
uv run --with matplotlib python -m kartstudy.trajectory_evolution \
  --input PATH_TO_RUN [ANOTHER_RUN ...] \
  --track zengarden --eval-seed 20000 \
  --output runs/trajectory-evolution-example
```

Inputs can be run directories, individual snapshot directories, or trusted local
`actor.pt`/`checkpoint.pt` files in this project's policy format. Run directories
with `snapshots/` use that saved history, sorted by the recorded action count;
otherwise they use the final actor/checkpoint. Different methods, training seeds
and backends may appear together, explicitly labeled. All inputs must use schema
3, 84 observations, five controls and the same episode horizon. `--steps 0 40000
160000 400000` restricts evaluation to existing snapshots at those counts. BC is
an explicitly fixed fitted reference; missing online snapshots remain missing.
No intermediate policy is reconstructed from a final checkpoint.

Each fresh output directory contains:

- `evolution.gif`: matched panels changing together every 800 ms (`--frame-ms`).
- `frames/actions-*.png`: every animation frame as a still image.
- `run-*-keyframes.png` and `.pdf`: one compact chronological strip per input.
  Choose its columns with `--key-steps 0 40000 160000 400000`; the default picks
  up to four chronological stages without ranking performance.
- Trajectory NPZs, cached native driveline, evaluation records and a provenance
  manifest with input/output hashes, local evaluation source, versions and platform.

Redraw or change the static strip without running the simulator again:

```sh
uv run --with matplotlib python -m kartstudy.trajectory_evolution \
  --render-only --output runs/trajectory-evolution-example \
  --key-steps 0 40000 160000 400000 --frame-ms 1200
```

Collection refuses an existing output directory. Rendering verifies cached input
hashes and may replace only its generated figures. These are fresh headless CPU
evaluations, including for CUDA-trained actors, and never replace official logs.
Any installed native race track is supported. An unfamiliar track means absent
from that run's training list; supplemental tracks such as Zen Garden are not
part of the original thirteen-track final assessment. Top-down maps can obscure
overpasses and dwell time, so use their saved episode metrics and trajectories
for interpretation. A single evaluation seed illustrates behavior, not robustness.
