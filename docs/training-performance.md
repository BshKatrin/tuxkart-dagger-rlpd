# Training performance

Training exports learner metrics only at the existing 32-action update boundary.
Losses, gradients, actor cadence, target equations, replay sampling, finite checks,
evaluation and checkpoint scheduling are unchanged. CUDA additionally groups
target-parameter multiply/add operations; CPU and Metal retain the original
per-parameter update. Differential checks compare
complete checkpoints, optimizer state, gradients and Python/Torch/device RNG.

## Measuring worker counts

Use fresh ignored output directories and record background campaigns. The update
benchmark uses frozen float32 training minibatches at batch 256 and hidden 128,
with the method's original ensemble and UTD. RLPD qualification batches use
completed SAC-sym online replay as an explicitly labelled timing proxy. Each NPZ requires an adjacent JSON
with `schema_version=1`, `phase="cached-training-update"`, `method`, `sha256` and
nonempty `sources`; arrays are `obs`, `next_obs`, `actions`, `reward`, `terminated`.
Exclude validation data and label proxy minibatches explicitly.

```sh
uv run --locked python scripts/benchmark_throughput.py \
  --baseline runs/bench/baseline_algorithms.py --batches runs/bench/sac.npz \
  --device mps --method sac --optimization metrics --workers 1 \
  --warmup-actions 64 --block-actions 128 --repeats 10 \
  --background runs/existing-study/manifest.json \
  --label mac-scalar-reporting --output runs/bench/mac-new
```

Use an archived baseline source, not a copy of the optimized implementation.
WSL commands additionally require host admission and the process-local CUDA
wrapper in [wsl.md](wsl.md). `--workers 2` means two **additional** fixture learners;
count existing learners when sizing a test. `--profile` records five untimed
post-measurement updates with one worker. Metal's available trace records CPU
dispatch rather than native GPU kernel time.

The benchmark refuses existing outputs and rejects nonfinite or unequal final
checkpoints, optimizers, gradients or RNG. `benchmark.json` records source/input
hashes, exact-state qualification, paired timing blocks and background snapshots.
Cached updates exclude rollout, inference, evaluation and initialization. Compare
whole native workload timings before choosing a suite worker count; shared-load
windows do not establish an idle-machine optimum or a full-study speedup.

## Scheduling and evaluation

`run_suite.py --queue-order long-first` schedules RLPD before SAC-sym and SAC,
without changing individual configurations. The default preserves the original
order. Replaying the same six measured tiny-job durations with
`benchmark_throughput.py --queue-replay durations.json --label scheduling
--output runs/bench/queue-new` estimates two-slot makespan; concurrent contention
and long-run scaling remain unmeasured.

`kartstudy.evaluation.evaluate(..., workers=2, worker_timeout=120)` optionally
partitions seeds across spawned CPU processes with private frozen actors and
headless simulators. It preserves input seed/trajectory order and leaves the
parent actor, environment and RNG untouched. Initialize shared assets first.
Training continues to use serial evaluation by default; startup can outweigh
parallelism for short assessments. Callable policies and rendered/GPU evaluation
are unsupported in this optional path. A failed worker is reported with bounded
teardown rather than leaving a partial evaluation.

## Qualification results

Matched update blocks used the original study shape and archived pre-change
learner. Ratios below are median paired speedups over that baseline, with 128
cached actions per block (10 SAC repeats, 5 RLPD repeats). Both hosts had two
unchanged RLPD learners running; the fixture count is additional to those jobs.

| Backend / method | One fixture | Two fixtures | Enabled change |
| --- | ---: | ---: | --- |
| Metal / SAC | 1.06× | 1.08× | Deferred metric reporting |
| Metal / RLPD | 1.01× | 1.00× | Deferred metric reporting |
| CUDA / SAC | 1.08× | 1.09× | Grouped targets + deferred metrics |
| CUDA / RLPD | 1.75× | 1.87× | Grouped targets + deferred metrics |

Each benchmark case qualified exact final checkpoint, optimizer, gradient and
RNG state. The qualification used Torch 2.15.0, Mac CPython 3.12.16 and WSL
CPython 3.12.14; runtime/input/source fingerprints are retained with the evidence.
Grouping targets alone gave no useful Metal gain, so Metal keeps its original
update. Metric deferral alone was nearly neutral on CUDA.

Native throughput fixtures additionally ran 256 actions across the four training
tracks, with evaluation disabled and warmup zero. Network/batch/UTD, controls,
replay sampling and snapshot/checkpoint behavior stayed intact. These shortened
fixtures cannot establish learning quality or full-campaign throughput. On the
Mac, running a matched two-seed workload concurrently improved the rollout/update
window by 1.54× for SAC and 1.67× for RLPD over sequential fixtures; final actor
hashes matched. The GPU-only SAC benchmark did not gain aggregate throughput
from an extra worker: native rollout/inference work changes the concurrency tradeoff.

A frozen final SAC seed-1 actor produced identical native Lighthouse records and
all trajectory arrays with serial and two-worker CPU evaluation. The initial Mac
measurement was 1.23 s serial versus 2.94 s parallel, including spawn/merge in the
evaluation call. This supports keeping the current four-seed assessment serial.
The same six measured Mac `train.run` fixture durations (process imports excluded) gave an estimated two-slot makespan
of 41.2 s in existing order versus 34.2 s with long-first scheduling; these are
queue estimates, not an observed full-study acceleration.

WSL native evaluation also preserved every record and trajectory array. Its
four-seed call took 4.29 s serial versus 5.50 s parallel under the recorded load,
so the same serial default is appropriate for this workload on both hosts.

The same shortened native workload on WSL improved the rollout/update window by
1.64× for SAC and 1.85× for RLPD with two fixture workers, with matching final actor
hashes. Whole supervised two-seed workloads took 9.84 s (SAC) and 17.65 s (RLPD),
versus 9.58 s and 23.83 s on the Mac in their respective recorded load windows.
These whole-workload timings include startup/checkpoint overhead; differences in
CPU simulation, interpreter/build and background load accompany the GPU change.
The WSL queue estimate was 34.7 s existing order versus 28.0 s long-first.

Both machines therefore have evidence for overlapping native work, but an idle
three/four-worker optimum and full-budget scaling still require a separate study.
The background campaigns retained their original code and worker count. For future runs,
select the worker count explicitly, retain available-memory headroom, and compare
full rollout/update throughput rather than GPU utilization alone.
