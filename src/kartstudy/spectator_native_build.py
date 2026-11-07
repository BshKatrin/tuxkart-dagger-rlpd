"""Compile a presentation adapter tied to the installed binary and verified source."""

import ctypes
import fcntl
import hashlib
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import pystk2

from kartstudy.native_telemetry import SOURCE_SHA256, SOURCE_URL

SUPPORTED_SIMULATOR_SHA256 = (
    "f66071e8a4c81bc6bca60c2120035a0065fd10a2f348b789dd98299300fc3207"
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load_adapter():
    if sys.platform != "darwin" or version("pysupertuxkart2") != "0.7.4":
        raise RuntimeError("Native spectator requires macOS and PySuperTuxKart2 0.7.4")
    root = Path(__file__).resolve().parents[2]
    cache = root / "runs/native-spectator"
    cache.mkdir(parents=True, exist_ok=True)
    adapter = Path(__file__).with_name("spectator_native.cpp")
    binary = Path(pystk2.__file__)
    fingerprint = {
        "simulator_version": version("pysupertuxkart2"),
        "simulator_sha256": _sha(binary),
        "source_sha256": SOURCE_SHA256,
        "adapter_sha256": _sha(adapter),
        "builder_sha256": _sha(Path(__file__)),
    }
    if fingerprint["simulator_sha256"] != SUPPORTED_SIMULATOR_SHA256:
        raise RuntimeError("Unsupported PySuperTuxKart2 binary fingerprint")
    build_key = hashlib.sha256(repr(fingerprint).encode()).hexdigest()[:24]
    library = cache / f"spectator-{build_key}.dylib"
    with (cache / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            source = cache / "source.tar.gz"
            if not source.exists():
                supplied = os.environ.get("KARTSTUDY_NATIVE_SOURCE_ARCHIVE")
                if supplied:
                    source.write_bytes(Path(supplied).read_bytes())
                else:
                    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
                        source.write_bytes(response.read())
            if _sha(source) != SOURCE_SHA256:
                raise RuntimeError("Simulator source checksum mismatch")
            # Always extract the verified archive; do not trust an existing header tree.
            with tempfile.TemporaryDirectory(prefix="headers-", dir=cache) as directory:
                headers = Path(directory)
                with tarfile.open(source) as archive:
                    archive.extractall(headers, filter="data")
                source_root = headers / "pysupertuxkart2-0.7.4"
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
                    "lib/graphics_engine/include",
                    "lib/glad/include",
                ):
                    command.extend(["-I", str(source_root / include)])
                temporary = library.with_suffix(".tmp.dylib")
                build = subprocess.run(
                    command + [str(adapter), "-o", str(temporary)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if build.returncode:
                    raise RuntimeError(
                        f"Native spectator build failed:\n{build.stderr}"
                    )
                temporary.replace(library)

    simulator = ctypes.CDLL(str(binary), mode=ctypes.RTLD_GLOBAL)
    native = ctypes.CDLL(str(library))
    native._simulator = simulator
    native.kartstudy_spectator_time.restype = ctypes.c_double
    native.kartstudy_spectator_error.restype = ctypes.c_char_p
    native.kartstudy_spectator_create.argtypes = [ctypes.c_int, ctypes.c_int]
    native.kartstudy_spectator_create.restype = ctypes.c_void_p
    float_p = ctypes.POINTER(ctypes.c_float)
    definitions = {
        "add": [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float],
        "camera": [ctypes.c_void_p, float_p, float_p, ctypes.c_float],
        "pose": [
            ctypes.c_void_p,
            ctypes.c_int,
            float_p,
            float_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
        ],
        "render": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)],
        "ghost_state": [ctypes.c_void_p, ctypes.c_int, float_p],
        "camera_state": [ctypes.c_void_p, float_p],
        "destroy": [ctypes.c_void_p],
        "test_fail_next_render": [ctypes.c_void_p],
        "release_players": [],
    }
    for name, arguments in definitions.items():
        function = getattr(native, f"kartstudy_spectator_{name}")
        function.argtypes = arguments
        function.restype = (
            None
            if name in ("destroy", "release_players", "test_fail_next_render")
            else ctypes.c_int
        )
    fingerprint["build_fingerprint"] = build_key
    return native, fingerprint
