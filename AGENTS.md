# Working on this project

This repository studies DAgger and RLPD for solo SuperTuxKart driving.
[README.md](README.md) explains the research questions;
[docs/initial-experiments.md](docs/initial-experiments.md) records the first experimental suite.

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

## Running experiments

Qualify the expert before collecting data. The collector requires every lap to
finish and refuses to overwrite an existing dataset.

```sh
uv run python scripts/qualify_expert.py
uv run python -m kartstudy.data --output data/expert.npz
uv run python -m kartstudy.data --output data/conservative.npz --conservative
uv run python scripts/run_suite.py --output runs/study --workers 2
```

The suite launcher generates configurations, bounds concurrency, saves a source
archive, and stops the queue if a worker fails. Use a fresh output directory
for each suite. Inspect the actual worker configuration and dataset metadata
before interpreting a result; the initial-experiments document describes the first suite’s intended protocol.

```sh
uv run python scripts/status.py runs/study
uv run python scripts/summarize.py runs/study
```

Generated datasets and runs are Git-ignored. Dataset metadata records the split,
expert provenance, and checksum. Each run records its configuration, resources,
training/evaluation logs, final trajectories, and rolling checkpoint. The suite's
`manifest.json` records worker PIDs and queue state. Summaries can include partial
runs; restrict final analysis to completed runs at matched budgets.

## Protect running work

Do not overwrite datasets or change experiment code under an active suite:
workers import code when they start, so queued jobs could otherwise use a
different implementation from running jobs. Documentation edits are independent
of training; preserve the suite's original source archive.

To stop launching queued jobs while letting active workers finish:

```sh
touch runs/study/STOP
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
