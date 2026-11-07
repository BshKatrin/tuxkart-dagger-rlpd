"""Run a command with PTX compiler libraries matching WSL's active CUDA driver."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

WSL_LIB = Path("/usr/lib/wsl/lib")
DRIVER_ROOT = Path("/usr/lib/wsl/drivers")
DRIVER_PROBE = """
import ctypes, json
from pathlib import Path
driver = ctypes.CDLL('/usr/lib/wsl/lib/libcuda.so.1')
code = driver.cuInit(0)
if code != 0:
    raise RuntimeError(f'CUDA driver initialization failed: {code}')
directories = set()
for line in Path('/proc/self/maps').read_text().splitlines():
    path = Path(line.split()[-1])
    if str(path).startswith('/usr/lib/wsl/drivers/') and path.name.startswith('libcuda.so'):
        directories.add(str(path.parent))
if len(directories) != 1:
    raise RuntimeError(f'Expected one active WSL driver directory, found {sorted(directories)}')
print(json.dumps({'driver_directory': directories.pop()}))
"""


def wsl_environment():
    if platform.system() != "Linux" or "microsoft" not in platform.release().lower():
        raise RuntimeError(
            "This wrapper requires WSL2; use ordinary uv commands elsewhere"
        )
    if not (WSL_LIB / "libcuda.so.1").is_file():
        raise RuntimeError("WSL CUDA driver library is unavailable")
    try:
        probe = subprocess.run(
            [sys.executable, "-c", DRIVER_PROBE],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(f"WSL CUDA driver probe failed: {detail}") from exc
    directory = Path(json.loads(probe.stdout)["driver_directory"]).resolve()
    if directory.parent != DRIVER_ROOT.resolve():
        raise RuntimeError("CUDA driver mapped outside the expected WSL driver root")
    if not (directory / "libnvidia-ptxjitcompiler.so.1").is_file():
        raise RuntimeError(f"Matching WSL PTX compiler is unavailable in {directory}")
    environment = os.environ.copy()
    paths = [str(WSL_LIB), str(directory)]
    paths += [
        p
        for p in environment.get("LD_LIBRARY_PATH", "").split(":")
        if p and p not in paths
    ]
    environment["LD_LIBRARY_PATH"] = ":".join(paths)
    # Retain the pinned Triton release's current fusion behavior explicitly.
    environment.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
    return environment, directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-environment", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        environment, directory = wsl_environment()
        if args.show_environment:
            print(
                json.dumps(
                    {
                        "driver_directory": str(directory),
                        "LD_LIBRARY_PATH": environment["LD_LIBRARY_PATH"],
                    },
                    indent=2,
                )
            )
            return
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("Supply a command after --")
        os.execvpe(command[0], command, environment)
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
