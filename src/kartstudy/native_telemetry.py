"""Read rescue animations from the pinned native simulator, without changing it."""

import ctypes
import fcntl
import hashlib
import subprocess
import sys
import tarfile
import urllib.request
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import pystk2

SOURCE_URL = "https://files.pythonhosted.org/packages/2f/29/1c1055e68dcf27f54429d6be928bfd9732e22087d65c108a012cf94d2626/pysupertuxkart2-0.7.4.tar.gz"
SOURCE_SHA256 = "4b806f9ed12400d3ee08f914abd2bc8114828aa8cd79ddd1bf2a015676c03819"


@lru_cache(maxsize=1)
def _library():
    if sys.platform != "darwin" or version("pysupertuxkart2") != "0.7.4":
        raise RuntimeError(
            "Native rescue telemetry requires macOS and PySuperTuxKart2 0.7.4"
        )
    cache = Path("runs/native-metrics").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    bridge = Path(__file__).with_suffix(".cpp")
    fingerprint = hashlib.sha256(
        Path(pystk2.__file__).read_bytes() + bridge.read_bytes()
    ).hexdigest()[:20]
    library = cache / f"telemetry-{fingerprint}.dylib"
    with (cache / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            source = cache / "source.tar.gz"
            if not source.exists():
                with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
                    source.write_bytes(response.read())
            if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
                raise RuntimeError("Simulator source checksum mismatch")
            with tarfile.open(source) as archive:
                archive.extractall(cache, filter="data")
            headers = cache / "pysupertuxkart2-0.7.4"
            command = [
                "xcrun",
                "clang++",
                "-std=c++17",
                "-shared",
                "-fPIC",
                "-undefined",
                "dynamic_lookup",
                "-DPYSTK_BUILD",
            ]
            for include in (
                "src",
                "lib/irrlicht/include",
                "lib/bullet/src",
                "lib/graphics_utils",
            ):
                command.extend(["-I", str(headers / include)])
            temporary = library.with_suffix(".tmp.dylib")
            subprocess.run(
                command + [str(bridge), "-o", str(temporary)],
                check=True,
                capture_output=True,
                text=True,
            )
            temporary.replace(library)
    # Export the already loaded simulator's symbols to this read-only bridge.
    simulator = ctypes.CDLL(pystk2.__file__, mode=ctypes.RTLD_GLOBAL)
    native = ctypes.CDLL(str(library))
    native._simulator = simulator  # Keep both handles alive together.
    native.kartstudy_rescue_time_left.argtypes = [ctypes.c_int]
    native.kartstudy_rescue_time_left.restype = ctypes.c_double
    return native


def rescue_time_left(kart_id=0):
    """Seconds remaining in a native rescue animation, or None otherwise."""
    value = _library().kartstudy_rescue_time_left(kart_id)
    return value if value >= 0 else None
