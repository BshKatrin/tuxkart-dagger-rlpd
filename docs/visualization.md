# Policy playback

Restore the release first: extract `selected-policies.zip` and `study-data.zip`
at the repository root, then run `uv sync --locked`.

```sh
uv run python -m kartstudy.playback
uv run python -m kartstudy.playback --policy rlpd --track stk_enterprise
uv run python -m kartstudy.playback --policy expert --track lighthouse
```

The default is DAgger. Select Policy and Track in the Tk window. The four learned
policies all retain **training seed 0**, alongside the original expert reference.
[Tracks and policies](tracks-and-policies.md) lists every CLI name, simulator track
ID, difficulty group and training/evaluation role. The viewer offers the 13 study
tracks; eight unused tracks are listed for reference only.

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

## Native ghost comparisons

The spectator shows frozen BC, DAgger, SAC and RLPD solo laps together in the
original textured SuperTuxKart scenery. Each ghost is a copied native kart model
without a physics body. Replay rendering changes camera and model transforms;
it runs neither the actors nor simulation steps. The engine's one hidden kart
loads the scenery. This is an **independent solo-lap replay**, with race-time
synchronization, rather than a new multi-car evaluation.

### Installation and recording

Use the existing Python 3.12 study environment (`uv sync --locked`) and restored
`polar-four-seed0-v1` actors. Install Xcode Command Line Tools if `xcrun clang++`
is unavailable. The presentation adapter supports the verified macOS
PySuperTuxKart2 0.7.4 binary with SHA-256
`f66071e8a4c81bc6bca60c2120035a0065fd10a2f348b789dd98299300fc3207`.
Other binaries fail explicitly. First render compiles a small adapter using the
checksum-verified 0.7.4 source archive. The archive and compiled libraries stay in
ignored `runs/native-spectator/`; temporary headers are removed after compilation.
The study wheel and lock are preserved.
`KARTSTUDY_NATIVE_SOURCE_ARCHIVE` can point to an already downloaded source archive.

```sh
uv run python -m kartstudy.spectator capture runs/visualizations/lighthouse-comparison \
  --track lighthouse --train-seed 0 --eval-seed 20000
uv run python -m kartstudy.spectator view runs/visualizations/lighthouse-comparison
uv run python -m kartstudy.spectator export runs/visualizations/lighthouse-comparison \
  runs/visualizations/lighthouse-comparison.mp4
```

Capture runs each selected actor sequentially with deterministic CPU inference
in the unmodified solo task. Defaults select the four released training-seed-0
actors, evaluation seed 20000, and 1500 actions per lap. Use `--methods bc dagger`
for a smaller comparison, `--track mines` for held-out Old Mine, or
`--root PATH` to select another completed study. Existing outputs are refused.
Recordings and exports belong under ignored `runs/visualizations/`.

Each recording directory contains a JSON manifest and checksum-protected compressed
arrays. They retain time zero, the terminal pose, world position, world-facing
`xyzw` orientation, speed, controls, and rescue/termination flags. The manifest
records source and simulator hashes, actor hashes and training counters, both
seeds, and each lap's completion status. These fresh laps are visualization
captures; released official evaluations remain authoritative.

### Camera and playback controls

- **Follow:** elevated behind the selected ghost, DAgger by default.
- **Corner:** fixed beside a chosen native track segment. Choose its segment index
  and adjust distance, height and orbit to frame the bend.
- **Overview:** frames the visible ghost group, zooming out as the group spreads.
- **Chase:** closer to the selected ghost for inspecting steering and recovery.

The viewer exposes play/pause (Space), single presentation step (Right arrow),
restart (R), timeline scrubbing, wall-time playback speed, ghost visibility,
follow target, camera mode, distance, height, orbit, FOV, and renderer reset.
The minimap locates visible ghosts outside the camera view. Finished, timed-out
and stopped laps hold their final pose and retain their status. Cameras never
switch automatically. A hidden follow target still anchors the follow camera.

Tk runs in the parent process and native rendering in a dedicated child. Loading
and frame requests run outside Tk's event loop; timeouts are bounded at 30 and
five seconds respectively. **Reset renderer** replaces only this viewer's child
and retains the replay time and settings. The earlier live-policy viewer remains
available with `python -m kartstudy.playback`.

Interactive defaults are 960×540. Export defaults are 1920×1080 at 30 fps, using
the same camera calculations and replay sampling as the viewer. Interactive
playback follows wall-time pacing; export samples fixed FPS intervals:

```sh
uv run python -m kartstudy.spectator export runs/visualizations/lighthouse-comparison \
  runs/visualizations/lighthouse-corner.mp4 --camera corner --segment 40 \
  --distance 15 --camera-height 6 --orbit 20 --start 0 --end 15
```

`--follow`, `--distance`, `--camera-height`, `--orbit`, `--fov`, `--segment`,
`--visible`, `--width` and `--height` work for both viewing and export. Orbit and
FOV are in degrees at the CLI; the native FOV boundary uses radians. Export's
`--start`/`--end` create an explicitly labeled partial preview. Every MP4 has a
JSON sidecar recording input checksums, camera settings, native/source fingerprints,
seeds, dimensions, frame count and time range. Preserve that sidecar with the video.

### Future learning evolution

New training runs export actor-only CPU snapshots at the existing assessment
and final/stop boundaries. The immutable `snapshots/actions-…_updates-…/`
directories include configuration, collected actions, optimizer updates, DAgger
round and source provenance. Repeated identical save boundaries deduplicate.
Logging preserves the live actor, RNG state and training/evaluation scheduling.
BC labels use fitting updates; online methods use collected actions.

```sh
uv run python -m kartstudy.spectator capture runs/visualizations/dagger-evolution \
  --root runs/my-future-study --methods dagger \
  --snapshots runs/my-future-study/shared_dagger_s0/snapshots --track lighthouse
```

Select all snapshots in one run, or pass individual snapshot directories after
`--snapshots`. Each snapshot drives a separate solo lap. These are successive
policy checkpoints, rather than genetic-algorithm generations. The existing
release contains final actors; historical snapshots cannot be reconstructed.

### Measurement limits

The capture interval is 0.1 seconds. Interpolated presentation frames add visual
smoothness without adding measurements. Quaternions follow the shortest rotation
path; rescue discontinuities hold the preceding pose until the next measurement.
Native orientation conversion is checked against identity and known engine turns.
Replay time follows the native race clock. Countdown actions and all duplicate
measurements remain on disk with a separate `action_time` clock. Rendering keeps
the initial zero-time pose and the final pose at each positive timestamp; a
zero-duration stopped lap displays its terminal pose/status. Format-1 native
recordings load read-only using their stored native timestamps. Position-only
analysis trajectories cannot supply native orientations.
The exact terminal endpoint is included in export, adding at most two frames of
terminal hold due to rounding to the requested frame rate.

Ghosts do not collide, collect items, or share race effects. Scenery and original
models are native; recorded dynamic track and kart effects are not replayed.
Steering poses are rendered, but wheel rotation and suspension motion are not
replayed. Native scene-animation callbacks can still run during rendering. Visibility is subject to
normal scenery occlusion. Use a closer, lower camera in tunnels, or select a
fixed corner for a diagnostic. Videos illustrate behavior; completion rates and
aggregate official evaluations support conclusions, with independent training
runs as the statistical unit.
