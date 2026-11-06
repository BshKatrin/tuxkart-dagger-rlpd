# Experiment artifacts

Datasets, runs, and packaged snapshots stay under ignored `data/`, `runs/`, and
`dist/`. Each suite stores configurations, a source archive, queue manifest,
training/evaluation logs, trajectories, and checkpoints. Dataset metadata records
training tracks, expert, schema, checksum, and training/validation split.

```sh
uv run python scripts/package_results.py --root runs/study-polar-four --output dist/study
```

Package only completed suites. Archives include the exact datasets referenced by
the suite and their metadata. Check `SHA256SUMS` before restoring; extract into a
fresh directory. Publish immutable snapshots as GitHub Release assets with a new
tag for each changed snapshot. Never overwrite an existing release snapshot.
Checkpoints include optimizer, replay, labels, counters, and RNG states, but
training resume is not implemented. See the [protocol](study.md).
