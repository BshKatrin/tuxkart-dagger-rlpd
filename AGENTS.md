# Working on this repository

This repository studies DAgger and RLPD for solo SuperTuxKart driving.
Read [README.md](README.md) for setup and usage and [docs/study.md](docs/study.md)
for the research protocol before changing experimental behavior.

## Preserve the comparison

- Keep the implementation small. Extend existing code and meaningful tests.
- Keep observations, controls, budgets, and evaluation conditions comparable.
  Distinguish symmetric sampling from RLPD's additional components; document adaptations.
- Keep executed actions separate from expert labels and validation data out of replay.
- Average evaluations within each training run before comparing methods. Independent
  training runs are the statistical unit. Always report completion beside conditional
  lap time; identify whether variability describes training runs or evaluation seeds.
- Inspect actual worker configurations and dataset metadata before interpreting results.
  Partial or stopped runs are not completed experiments.
- Preserve protocols and source snapshots used by completed runs. Record supplemental
  experiments in a separate protocol, linked from the relevant study document.

## Where to work

- `src/kartstudy/`: task wrapper, learning, evaluation, playback, and diagnostics.
- `scripts/`: experiment launchers, summaries, and release packaging.
- `report/scripts/`: analysis and figure generation; outputs go in ignored `report/generated/`.
- `tests/`: learning semantics, terminal handling, data separation, statistical
  accounting, and native simulator behavior.

Use Python 3.12 and `uv` for commands. Training explicitly requires PyTorch Metal
(`mps`); simulation and rollout/evaluation inference use CPU. Keep one native
simulator per worker process. Bound concurrent learners using measured throughput;
more Metal processes do not necessarily train faster.

Rescue telemetry uses a read-only native bridge for PySuperTuxKart2 0.7.4.
First use downloads checksum-verified source headers and compiles with
`xcrun clang++`. Generated files belong in ignored `runs/native-metrics/`.
Rebuilds are keyed to the simulator binary and bridge source. Unsupported versions
must fail explicitly rather than silently report zero rescues.

Keep Tk and SDL/OpenGL in separate processes on macOS: combining their Cocoa
application objects crashes. Playback must use the shared task wrapper and
CPU deterministic inference at the original 0.1-second action interval. Speed
controls affect wall time only. Rendered replay may differ slightly from headless
evaluation; never replace official logs with video metadata.

## Protect running work

Before editing experiment code, check for active suites. Workers import code when
they start; queued jobs must not silently use a different implementation from
running jobs. Do not overwrite datasets. Documentation changes are independent.

To stop queued launches while active workers finish, create `STOP` in the suite
root. To stop active workers too, send SIGTERM only to the suite PID recorded in
its manifest. Workers save at a safe boundary. Check individual `status.json`
files afterward. Checkpoints retain model, optimizer, replay/labels, counters,
and RNG state, but automatic resume is not implemented; rerun in a fresh directory.

For detached macOS jobs, `caffeinate -i -w <suite PID>` inhibits idle sleep for
that process's lifetime. Closing the laptop can still suspend computation.

## Verify changes

Run focused checks while editing and the relevant existing tests before long jobs:

```sh
uv run ruff check src scripts tests
uv run pytest -q
```

Extend tests for meaningful behavior or plausible failures, not implementation
mirrors. Check changed analysis scripts too. Documentation-only changes need
link/content checks, not training runs.

## Documentation and releases

README serves readers: purpose, setup, usage, and routes to results. This file
serves agents: constraints, verification, and implementation traps. Put scientific
choices in the relevant protocol and artifact restoration in
[docs/experiment-data.md](docs/experiment-data.md). Update affected guidance when
behavior changes; avoid duplicate explanations and file-by-file implementation catalogs.

Never commit live PIDs, estimates, progress reports, temporary plans, or generated
artifacts. Execution state belongs in run artifacts. Keep `data/`, `runs/`, and
`dist/` out of Git history. Publish completed, checksum-verified snapshots with
`scripts/package_results.py`, preserving original run files and source archives.
Use a new release tag for changed data; never overwrite a published snapshot.
