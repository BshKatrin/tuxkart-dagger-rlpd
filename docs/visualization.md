# Policy playback

```sh
uv run python -m kartstudy.playback --policy expert
uv run python -m kartstudy.playback --root runs/study-polar-v3 --policy dagger --track zengarden
```

Select the track, policy, and training seed. Only completed compatible checkpoints
are offered. When no checkpoint exists for a selection, the viewer uses the
original expert. Playback always uses normal driving, deterministic CPU inference,
and the shared 0.1-second action interval. Pause, step, restart, and speed controls
change wall time only. Tk and the simulator run in separate processes on macOS.

```sh
uv run python -m kartstudy.playback --policy expert --track zengarden \
  --record runs/visualizations/expert-zengarden.mp4
```

Recording saves an MP4 and JSON sidecar and refuses existing output paths.
`--limit-steps 100` records a partial preview. Fresh rendering can cause tiny
floating-point differences; official evaluation logs remain authoritative.
