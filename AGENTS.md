# Working on this project

This repository studies DAgger and RLPD for solo SuperTuxKart driving.
[README.md](README.md) explains the research questions;
[docs/study.md](docs/study.md) defines the study protocol.

## Working principles

- Keep the implementation small. Prefer extending existing code and tests over adding machinery.
- Preserve comparable observations, controls, and evaluation conditions across methods.
- Distinguish symmetric sampling from the additional RLPD components. Describe adaptations explicitly.
- Keep executed actions separate from expert labels, and validation data out of training replay.
- Treat independent training runs as the statistical unit. Always report completion beside conditional lap time.
- Document experimental changes in the relevant study protocol; preserve the protocol used by completed runs. Keep the README about the project; put development guidance here.
- Do not put live PIDs, runtime estimates, progress reports, or temporary plans in tracked documentation. Run artifacts record execution state.

## Environment

Use Python 3.12 and `uv` for dependencies and commands:

```sh
uv sync --locked
```

Training uses PyTorch Metal (`mps`) explicitly; it errors if unavailable.
Simulation and rollout/evaluation inference run on CPU. Keep one native simulator
per worker process, rather than sharing it between threads. The simulator
downloads its assets on first use. Limit concurrent learners based on measured
throughput; more Metal processes do not necessarily train faster.

Rescue telemetry uses a read-only native bridge against PySuperTuxKart2 0.7.4.
First use downloads checksum-verified source headers and compiles the small bridge
with Xcode Command Line Tools (`xcrun clang++`). Generated files stay in ignored
`runs/native-metrics/`; the simulator and policy are not modified. Rebuild is keyed
to the installed simulator binary and bridge source. Unsupported versions fail
explicitly rather than silently reporting zero rescues.

## Running experiments

Qualify the expert before collecting data. The collector requires every lap to
finish and refuses to overwrite an existing dataset.

```sh
uv run python scripts/qualify_expert.py --output runs/qualification-polar-v3/metrics.json
uv run python -m kartstudy.data --output data/team4-polar-v3/demonstrations.npz
uv run python scripts/run_suite.py --output runs/study-polar-v3 --workers 2
```

The suite launcher generates configurations, bounds concurrency, saves a source
archive, and stops the queue if a worker fails. Use a fresh output directory
for each suite. Inspect the actual worker configuration and dataset metadata
before interpreting a result; the study protocol describes the intended comparison.

```sh
uv run python scripts/status.py runs/study-polar-v3
uv run python scripts/summarize.py runs/study-polar-v3
```

Generated datasets and runs are Git-ignored. Dataset metadata records the split,
expert provenance, and checksum. Each run records its configuration, resources,
training/evaluation logs, final trajectories, and rolling checkpoint. The suite's
`manifest.json` records worker PIDs and queue state. Summaries can include partial
runs; restrict final analysis to completed runs at matched budgets.

## Policy visualization

`uv run python -m kartstudy.playback` opens the native viewer; see
[docs/visualization.md](docs/visualization.md) for controls and recording.
Playback uses the shared task wrapper and deterministic CPU inference at the
original 0.1-second action interval. Speed controls affect wall time only.
Keep Tk and the SDL/OpenGL simulator in separate processes on macOS; combining
their Cocoa application objects in one process crashes. The viewer discovers
completed checkpoints from run metadata. Outputs belong under ignored `runs/`.
Fresh rendering can introduce tiny floating-point differences from headless
evaluation. Preserve official evaluation logs; never replace them with video metadata.

## Sharing experiment artifacts

Keep `data/`, `runs/`, and `dist/` out of Git history. Publish completed snapshots
as GitHub Release assets using `scripts/package_results.py`; restoration and
archive contents are documented in [docs/experiment-data.md](docs/experiment-data.md).
Preserve the original run files and source archive. Do not overwrite a published
snapshot; use a new release tag for changed data.

## Protect running work

Do not overwrite datasets or change experiment code under an active suite:
workers import code when they start, so queued jobs could otherwise use a
different implementation from running jobs. Documentation edits are independent
of training; preserve the suite's original source archive.

To stop launching queued jobs while letting active workers finish:

```sh
touch runs/study-polar-v3/STOP
```

To stop active workers too, send SIGTERM to the suite PID recorded in its
manifest. Workers save at a safe boundary. Checkpoints contain model, optimizer,
replay/labels, counters, and RNG state, but automatic training resume is not
implemented. Check each worker's `status.json`; a stopped run is not a completed
experiment. Rerun interrupted experiments in a fresh directory.

For detached macOS runs, `caffeinate -i -w <suite PID>` inhibits idle sleep for
that process's lifetime. Closing the laptop can still suspend computation.

## Verification

Use focused checks while editing, and run the relevant existing tests before
launching long jobs:

```sh
uv run ruff check src scripts tests
uv run pytest -q
```

Tests should protect learning semantics, terminal handling, data separation,
statistical accounting, and real simulator behavior. Extend existing tests when
possible. Documentation-only edits need link/content checks, not training runs.
