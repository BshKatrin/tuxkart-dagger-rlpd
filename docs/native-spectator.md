# Native spectator camera preview

The optional spectator renders original SuperTuxKart scenery and visual-only kart
ghosts from supplied poses. Camera placement and rendering do not advance physics
or run a policy. Existing training and evaluation paths do not use this adapter.

Use Python 3.12, `uv sync --locked`, and Xcode Command Line Tools on macOS. The
adapter supports the checksum-pinned PySuperTuxKart2 0.7.4 Mac binary only; Linux
and other binaries fail explicitly. First use verifies downloaded source headers
and compiles the adapter. Builds stay under ignored `runs/native-spectator/`.

Render a static example at 960×540; output PNGs and their JSON metadata must be new:

```sh
uv run python scripts/preview_spectator.py runs/visualizations/native-follow.png --ghosts 1
uv run python scripts/preview_spectator.py runs/visualizations/native-corner.png --camera corner
uv run python scripts/preview_spectator.py runs/visualizations/native-overview.png --track mines --camera overview
```

These examples are labeled static pose fixtures. They demonstrate cameras and
ghosts, not trained-policy performance or a measured race. This initial port
provides the native API and preview; the interactive replay viewer is separate.

`kartstudy.spectator_native.NativeSpectator` is a context manager exposing
`geometry()`, `set_camera()`, `set_ghost_poses()`, `render()` and `diagnostics()`.
Camera positions/targets use world coordinates, with field of view in radians.
Ghost rotations are unit quaternions ordered `x, y, z, w`. They differ from the
simulator's exported inverse `w, x, y, z` convention. Replacing a pose set hides
omitted ghosts, so seeking replaces rather than accumulates presentation state.
One spectator owns its simulator process/thread; keep it separate from Tk and
other native simulators and always close it, including after rendering errors.

Native acceptance tests render Lighthouse, STK Enterprise and Old Mine, checking
orientation, pose replacement, unchanged physical kart count/clock, repeated
reloads and failure teardown. Retain their frames and JSON evidence by setting
`KARTSTUDY_NATIVE_PROOF_DIR` to a fresh ignored directory:

```sh
KARTSTUDY_NATIVE_PROOF_DIR=runs/visualizations/native-acceptance uv run pytest -q tests/test_spectator_native.py
```
