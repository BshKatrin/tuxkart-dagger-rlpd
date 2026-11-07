"""Read rescue animations from the pinned native simulator, without changing it."""

import ctypes
import fcntl
import hashlib
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import pystk2

SOURCE_URL = "https://files.pythonhosted.org/packages/2f/29/1c1055e68dcf27f54429d6be928bfd9732e22087d65c108a012cf94d2626/pysupertuxkart2-0.7.4.tar.gz"
SOURCE_SHA256 = "4b806f9ed12400d3ee08f914abd2bc8114828aa8cd79ddd1bf2a015676c03819"
BUILD_TIMEOUT = 60
# PySTK 0.7.4 cp312 manylinux_2_34 x86_64. Other layouts need native qualification.
SUPPORTED_LINUX_SIMULATOR_SHA256 = (
    "69e8d9a97033dafe96a634693e2e1246ab468b12ddcea58fb5dc2570da118b68"
)
# The pinned engine function, rather than bundled libraries, establishes string ABI.
LINUX_STRING_SYMBOLS = {
    0: "_ZN5World10createKartERKSsiiiN11RaceManager8KartTypeE13HandicapLevel",
    1: "_ZN5World10createKartERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEiiiN11RaceManager8KartTypeE13HandicapLevel",
}


def _linux_string_abi(simulator):
    found = []
    for abi, symbol in LINUX_STRING_SYMBOLS.items():
        try:
            getattr(simulator, symbol)
        except AttributeError:
            continue
        found.append(abi)
    if len(found) != 1:
        raise RuntimeError("Unsupported or ambiguous PySTK libstdc++ string ABI")
    try:
        _ = simulator._ZN5World7m_worldE
    except AttributeError as exc:
        raise RuntimeError("PySTK does not export the native World symbol") from exc
    return found[0]


def _validate_linux_binary(binary):
    if (
        platform.machine() != "x86_64"
        or platform.python_implementation() != "CPython"
        or sys.version_info[:2] != (3, 12)
    ):
        raise RuntimeError("Linux native telemetry requires CPython 3.12 on x86_64")
    if (
        hashlib.sha256(binary.read_bytes()).hexdigest()
        != SUPPORTED_LINUX_SIMULATOR_SHA256
    ):
        raise RuntimeError("Unsupported Linux PySuperTuxKart2 binary fingerprint")


def _build_key(binary, bridge, abi):
    digest = hashlib.sha256()
    for value in (
        binary.read_bytes(),
        bridge.read_bytes(),
        Path(__file__).read_bytes(),
        SOURCE_SHA256.encode(),
        f"{sys.platform}:0.7.4:{abi}".encode(),
    ):
        digest.update(value)
    return digest.hexdigest()[:20]


def _lock_build(lock):
    deadline = time.monotonic() + BUILD_TIMEOUT
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for native telemetry build lock")
            time.sleep(0.05)


def _source_archive(cache):
    source = cache / "source.tar.gz"
    if not source.exists():
        temporary = source.with_suffix(".tmp")
        try:
            supplied = os.environ.get("KARTSTUDY_NATIVE_SOURCE_ARCHIVE")
            if supplied:
                temporary.write_bytes(Path(supplied).read_bytes())
            else:
                with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
                    temporary.write_bytes(response.read())
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != SOURCE_SHA256:
                raise RuntimeError("Simulator source checksum mismatch")
            temporary.replace(source)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Simulator source checksum mismatch")
    return source


def _run_compile(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=BUILD_TIMEOUT)
    except BaseException:
        # The compiler driver can have children; stop its owned group before cleanup.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=5)
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _compile(cache, library, bridge, abi):
    if sys.platform == "darwin":
        command = ["xcrun", "clang++", "-undefined", "dynamic_lookup"]
    else:
        compiler = shutil.which("c++")
        if compiler is None:
            raise RuntimeError("Linux native telemetry requires a C++17 compiler")
        command = [compiler, f"-D_GLIBCXX_USE_CXX11_ABI={abi}"]
    source = _source_archive(cache)
    command += ["-std=c++17", "-shared", "-fPIC", "-DPYSTK_BUILD"]
    temporary = library.with_name(library.stem + ".tmp" + library.suffix)
    try:
        # Never use an unverified existing header tree. Keep only the pinned archive.
        with tempfile.TemporaryDirectory(prefix="headers-", dir=cache) as directory:
            headers = Path(directory)
            with tarfile.open(source) as archive:
                archive.extractall(headers, filter="data")
            root = headers / "pysupertuxkart2-0.7.4"
            for include in (
                "src",
                "lib/irrlicht/include",
                "lib/bullet/src",
                "lib/graphics_utils",
            ):
                command.extend(["-I", str(root / include)])
            build = _run_compile(command + [str(bridge), "-o", str(temporary)])
            if build.returncode:
                raise RuntimeError(
                    f"Native telemetry build failed:\n{build.stderr[-10000:]}"
                )
            temporary.replace(library)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Native telemetry build exceeded 60 seconds") from exc
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def _library():
    if sys.platform not in ("darwin", "linux") or version("pysupertuxkart2") != "0.7.4":
        raise RuntimeError(
            "Native rescue telemetry requires macOS or Linux and PySuperTuxKart2 0.7.4"
        )
    binary = Path(pystk2.__file__)
    if sys.platform == "linux":
        _validate_linux_binary(binary)
    # Export the already loaded simulator's symbols to this read-only bridge.
    simulator = ctypes.CDLL(str(binary), mode=ctypes.RTLD_GLOBAL)
    abi = _linux_string_abi(simulator) if sys.platform == "linux" else None
    cache = Path("runs/native-metrics").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    bridge = Path(__file__).with_suffix(".cpp")
    fingerprint = _build_key(binary, bridge, abi)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    library = cache / f"telemetry-{fingerprint}{suffix}"
    with (cache / "build.lock").open("w") as lock:
        _lock_build(lock)
        if not library.exists():
            _compile(cache, library, bridge, abi)
    native = ctypes.CDLL(str(library))
    native._simulator = simulator  # Keep both handles alive together.
    if abi is not None:
        native.kartstudy_telemetry_cpp_string_abi.argtypes = []
        native.kartstudy_telemetry_cpp_string_abi.restype = ctypes.c_int
        if native.kartstudy_telemetry_cpp_string_abi() != abi:
            raise RuntimeError(
                "Native telemetry compiler does not match PySTK string ABI"
            )
    native.kartstudy_rescue_time_left.argtypes = [ctypes.c_int]
    native.kartstudy_rescue_time_left.restype = ctypes.c_double
    return native


def rescue_time_left(kart_id=0):
    """Seconds remaining in a native rescue animation, or None otherwise."""
    value = _library().kartstudy_rescue_time_left(kart_id)
    if not math.isfinite(value):
        raise RuntimeError("Native rescue telemetry returned a nonfinite timer")
    return value if value >= 0 else None
