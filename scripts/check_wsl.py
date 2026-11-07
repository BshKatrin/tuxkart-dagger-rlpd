"""Verify WSL CUDA and native telemetry using bounded disposable fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from wsl_run import wsl_environment

GPU_PROBE = """
import hashlib, json, sys, sysconfig
from importlib.metadata import version
from pathlib import Path
if not (Path(sysconfig.get_path('include')) / 'Python.h').is_file():
    raise RuntimeError('Python.h is missing; use a uv-managed Python 3.12 interpreter')
import torch, pystk2
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable; CPU fallback is not a readiness pass')
assert torch.ones(1, device='cuda').cpu().item() == 1
libraries = set()
for line in Path('/proc/self/maps').read_text().splitlines():
    path = Path(line.split()[-1])
    if path.name.startswith(('libcuda.so', 'libnvidia-ptxjitcompiler.so')):
        libraries.add(path.resolve())
print(json.dumps({
    'python': sys.version, 'gpu': torch.cuda.get_device_name(0),
    'versions': {p: version(p) for p in ('torch', 'pysupertuxkart2', 'rl-mind', 'numpy')},
    'cuda_runtime': torch.version.cuda,
    'simulator_sha256': hashlib.sha256(Path(pystk2.__file__).read_bytes()).hexdigest(),
    'libraries': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(libraries)},
}))
"""


def stop_owned(process):
    # SIGINT lets pytest unwind the bridge's separately owned compiler group.
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        return process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.communicate(timeout=5)


def checked(command, environment, output, timeout):
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = stop_owned(process)
        output.write_text(stdout + stderr + f"\nTimed out after {timeout} seconds\n")
        raise RuntimeError(f"Readiness check timed out; see {output}") from exc
    except BaseException:
        stdout, stderr = stop_owned(process)
        output.write_text(stdout + stderr)
        raise
    output.write_text(stdout + stderr)
    if process.returncode:
        raise RuntimeError(f"Readiness check exited {process.returncode}; see {output}")
    return stdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {
        "status": "failed",
        "purpose": "small readiness fixtures; no training campaign or study evaluation",
    }
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(str(exc))
    try:
        environment, directory = wsl_environment()
        environment["KARTSTUDY_RUN_NATIVE_TELEMETRY_TESTS"] = "1"
        metadata = checked(
            [sys.executable, "-c", GPU_PROBE], environment, args.output / "cuda.log", 30
        )
        report["runtime"] = json.loads(metadata)
        compiler = [
            p
            for p in report["runtime"]["libraries"]
            if Path(p).name.startswith("libnvidia-ptxjitcompiler.so")
        ]
        driver = [
            p
            for p in report["runtime"]["libraries"]
            if Path(p).name.startswith("libcuda.so") and Path(p).parent == directory
        ]
        if (
            not driver
            or not compiler
            or any(Path(p).parent != directory for p in compiler)
        ):
            raise RuntimeError(
                "Loaded CUDA driver and PTX compiler do not match the active WSL directory"
            )
        print(
            "CUDA allocation and matching driver libraries passed; checking native rescue and checkpoint fixtures",
            flush=True,
        )
        checked(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"--junitxml={args.output / 'fixtures.xml'}",
                "tests/test_native_telemetry.py::test_real_rescue_activation_readonly_clock_and_reset",
                "tests/test_snapshots.py",
                "-k",
                "cuda or real_rescue_activation",
            ],
            environment,
            args.output / "fixtures.log",
            180,
        )
        cases = ET.parse(args.output / "fixtures.xml").findall(".//testcase")
        if any(
            case.find(tag) is not None
            for case in cases
            for tag in ("skipped", "failure", "error")
        ):
            raise RuntimeError("Required readiness fixtures were skipped or failed")
        if (
            not any(
                case.attrib["name"]
                == "test_real_rescue_activation_readonly_clock_and_reset"
                for case in cases
            )
            or sum("cuda" in case.attrib["name"] for case in cases) < 5
        ):
            raise RuntimeError("Required native or CUDA fixtures did not run")
        report["driver_directory"] = str(directory)
        report["fixtures"] = (
            "native rescue activation/reset/read-only clock; four-action mocked CUDA training and actor snapshot/RNG/checkpoint checks"
        )
        report["source_sha256"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("wsl_run.py"),
                *Path("src/kartstudy").glob("*.py"),
                Path("src/kartstudy/native_telemetry.cpp"),
                Path("tests/test_snapshots.py"),
                Path("tests/test_native_telemetry.py"),
                Path("uv.lock"),
            ]
        }
        report["status"] = "passed"
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        parser.exit(1, f"{exc}\n")
    finally:
        (args.output / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"WSL readiness passed: {args.output / 'readiness.json'}", flush=True)


if __name__ == "__main__":
    main()
