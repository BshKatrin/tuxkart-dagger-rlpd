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
