# WSL CUDA training

The headless simulator and learners support WSL2 with NVIDIA CUDA, CPython 3.12
on x86_64, and the pinned PySuperTuxKart2 0.7.4 Linux wheel. Metal remains the
default for the original Mac study. Select CUDA explicitly; unavailable devices
fail instead of falling back to CPU.

## Installation and readiness

Use the repository's unchanged `uv.lock`. An isolated managed interpreter supplies
the Python headers needed by Triton; a system Python without `Python.h` is not
sufficient. Run installation under the host's `devbox-storage` admission with an
estimate covering dependency/cache growth and the simulator's first-use assets.
The host's Windows and Linux storage reserves remain enforced.

```sh
uv python install 3.12.14 --no-bin --install-dir runs/wsl-python
uv sync --locked --python runs/wsl-python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12
uv run --locked python scripts/check_wsl.py --output runs/wsl-readiness-new
```

The readiness command chooses libraries matching the active Windows/WSL CUDA
driver, then checks GPU allocation and executes small fixtures: a native rescue
event over 100 simulator actions, and four-action mocked training fixtures for
actor snapshots, CUDA RNG preservation, checkpoint state and CPU reloads. It does
not launch a study or use its demonstration dataset. Required checks cannot be
skipped. Each subprocess is bounded; existing output directories are refused.
Retain the resulting JSON, fixture logs, JUnit XML and library/source checksums.

The simulator assets initialize on first use. Native telemetry compiles with a
C++17 compiler (`c++`) against checksum-verified headers. Linux additionally
checks the installed simulator binary and exported C++ string ABI. Unsupported
binaries fail explicitly. Generated libraries and the verified source archive
stay in ignored `runs/native-metrics/`; temporary headers are removed.
`KARTSTUDY_NATIVE_SOURCE_ARCHIVE` can reuse a verified source archive.

## CUDA library selection

A distribution-installed NVIDIA PTX compiler can shadow the compiler shipped
with the current Windows driver. This caused a CUDA-initialization crash with
the original package setup. `scripts/wsl_run.py` derives the active driver
directory from an isolated native-driver probe and places its matching libraries
first for the child process. It does not change system drivers or global loader
configuration, and does not hardcode a driver-directory identifier.
It explicitly retains the pinned Triton release's current floating-point fusion
default unless the caller already selected one.

Use the wrapper whenever invoking CUDA code:

```sh
uv run --locked python scripts/wsl_run.py --show-environment
uv run --locked python scripts/wsl_run.py -- python -c \
  'import torch; print(torch.cuda.get_device_name(0))'
```

After a Windows driver update, rerun readiness in a fresh output directory.
Do not replace system NVIDIA packages or set a persistent global library path
as a workaround for this project.

## Future experiments

Restore the verified demonstration dataset as described in
[experiment-data.md](experiment-data.md), then qualify the expert with the exact
Linux source/environment before launching a suite:

```sh
uv run --locked python scripts/wsl_run.py -- python scripts/qualify_expert.py \
  --output runs/qualification-wsl-new/metrics.json
uv run --locked python scripts/wsl_run.py -- python scripts/run_suite.py \
  --device cuda --methods sac sac_sym rlpd --seeds 1 2 --workers 1 \
  --qualification runs/qualification-wsl-new/metrics.json \
  --output runs/study-wsl-new
```

Heavy commands require host admission. Measure complete rollout/update throughput
and available GPU memory before raising worker count; tiny readiness fixtures are
not throughput estimates. Keep one native simulator per process. CUDA RNG states
are saved with checkpoints and restored after assessment; actor-only snapshots
remain portable to CPU and the Mac spectator.

Treat Linux/CUDA as a separate backend for study reporting. Preserve the original
Mac runs and source archives; the comparison command rejects mixing Metal and
CUDA training runs. Numerical differences can also arise from the simulator
build, so readiness does not replace expert qualification or final evaluation.
