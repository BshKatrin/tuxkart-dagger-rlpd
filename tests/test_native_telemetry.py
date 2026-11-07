import hashlib
import io
import os
import signal
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from kartstudy import native_telemetry as telemetry


class Function:
    def __init__(self, value):
        self.value = value

    def __call__(self, *args):
        return self.value


def simulator(abis=(1,), world=True):
    symbols = {telemetry.LINUX_STRING_SYMBOLS[abi]: Function(None) for abi in abis}
    if world:
        symbols["_ZN5World7m_worldE"] = Function(None)
    return SimpleNamespace(**symbols)


@pytest.fixture
def build_fixture(monkeypatch, tmp_path):
    telemetry._library.cache_clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(telemetry.sys, "platform", "linux")
    monkeypatch.setattr(telemetry.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(telemetry.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(telemetry, "version", lambda name: "0.7.4")
    binary = tmp_path / "pystk2.so"
    binary.write_bytes(b"simulator-binary")
    monkeypatch.setattr(
        telemetry,
        "SUPPORTED_LINUX_SIMULATOR_SHA256",
        hashlib.sha256(binary.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(telemetry.pystk2, "__file__", str(binary))
    content = io.BytesIO()
    with tarfile.open(fileobj=content, mode="w:gz") as archive:
        member = tarfile.TarInfo("pysupertuxkart2-0.7.4/src/verified.hpp")
        member.size = 8
        archive.addfile(member, io.BytesIO(b"verified"))
    source = content.getvalue()
    monkeypatch.setattr(telemetry, "SOURCE_SHA256", hashlib.sha256(source).hexdigest())
    monkeypatch.delenv("KARTSTUDY_NATIVE_SOURCE_ARCHIVE", raising=False)
    monkeypatch.setattr(
        telemetry.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(source)
    )
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: "/usr/bin/c++")
    native = SimpleNamespace(
        kartstudy_telemetry_cpp_string_abi=Function(1),
        kartstudy_rescue_time_left=Function(-1.0),
    )
    original = simulator()
    monkeypatch.setattr(
        telemetry.ctypes,
        "CDLL",
        lambda path, **kwargs: original if Path(path) == binary else native,
    )
    commands = []

    def compile(command):
        commands.append(command)
        include = Path(command[command.index("-I") + 1])
        assert (include / "verified.hpp").read_bytes() == b"verified"
        Path(command[-1]).write_bytes(b"bridge")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(telemetry, "_run_compile", compile)
    yield SimpleNamespace(
        binary=binary,
        native=native,
        simulator=original,
        commands=commands,
        cache=tmp_path / "runs/native-metrics",
        source=source,
    )
    telemetry._library.cache_clear()


@pytest.mark.parametrize("abi", [0, 1])
def test_linux_abi_uses_exact_engine_export(abi):
    assert telemetry._linux_string_abi(simulator((abi,))) == abi


@pytest.mark.parametrize("abis", [(), (0, 1)])
def test_unknown_or_mixed_linux_string_abi_fails_closed(abis):
    with pytest.raises(RuntimeError, match="Unsupported or ambiguous"):
        telemetry._linux_string_abi(simulator(abis))


def test_missing_world_export_fails_closed():
    with pytest.raises(RuntimeError, match="World symbol"):
        telemetry._linux_string_abi(simulator(world=False))


@pytest.mark.parametrize("platform,version", [("win32", "0.7.4"), ("linux", "0.7.5")])
def test_unsupported_platform_or_version_rejected_before_build(
    build_fixture, monkeypatch, platform, version
):
    monkeypatch.setattr(telemetry.sys, "platform", platform)
    monkeypatch.setattr(telemetry, "version", lambda name: version)
    with pytest.raises(RuntimeError, match="requires macOS or Linux"):
        telemetry._library()
    assert not build_fixture.cache.exists()
    assert not build_fixture.commands


def test_linux_build_cache_fingerprint_and_temporary_header_cleanup(
    build_fixture, monkeypatch
):
    native = telemetry._library()
    assert native is build_fixture.native
    assert native._simulator is build_fixture.simulator
    command = build_fixture.commands[0]
    assert command[:2] == ["/usr/bin/c++", "-D_GLIBCXX_USE_CXX11_ABI=1"]
    assert "-shared" in command and "-fPIC" in command
    assert "dynamic_lookup" not in command
    assert list(build_fixture.cache.glob("telemetry-*.so"))
    assert not list(build_fixture.cache.glob("headers-*"))
    assert not list(build_fixture.cache.glob("*.tmp*"))
    assert telemetry.rescue_time_left() is None
    assert len(build_fixture.commands) == 1
    telemetry._library.cache_clear()
    telemetry._library()
    assert len(build_fixture.commands) == 1
    build_fixture.binary.write_bytes(b"new-simulator-binary")
    # A separately qualified replacement changes the build key, even at one version.
    monkeypatch.setattr(
        telemetry,
        "SUPPORTED_LINUX_SIMULATOR_SHA256",
        hashlib.sha256(build_fixture.binary.read_bytes()).hexdigest(),
    )
    telemetry._library.cache_clear()
    telemetry._library()
    assert len(build_fixture.commands) == 2


def test_mac_keeps_xcrun_dynamic_lookup_and_dylib(build_fixture, monkeypatch):
    monkeypatch.setattr(telemetry.sys, "platform", "darwin")
    telemetry._library()
    command = build_fixture.commands[0]
    assert command[:4] == ["xcrun", "clang++", "-undefined", "dynamic_lookup"]
    assert not any("GLIBCXX" in part for part in command)
    assert list(build_fixture.cache.glob("telemetry-*.dylib"))
    assert not list(build_fixture.cache.glob("headers-*"))


def test_linux_old_string_abi_sets_matching_compiler_flag(build_fixture, monkeypatch):
    original = simulator((0,))
    build_fixture.native.kartstudy_telemetry_cpp_string_abi.value = 0
    monkeypatch.setattr(
        telemetry.ctypes,
        "CDLL",
        lambda path, **kwargs: (
            original if Path(path) == build_fixture.binary else build_fixture.native
        ),
    )
    telemetry._library()
    assert "-D_GLIBCXX_USE_CXX11_ABI=0" in build_fixture.commands[0]


def test_bridge_and_headers_hashes_change_build_key(
    build_fixture, tmp_path, monkeypatch
):
    bridge = tmp_path / "bridge.cpp"
    bridge.write_bytes(b"old-source")
    old = telemetry._build_key(build_fixture.binary, bridge, 1)
    bridge.write_bytes(b"new-source")
    changed = telemetry._build_key(build_fixture.binary, bridge, 1)
    assert changed != old
    assert telemetry._build_key(build_fixture.binary, bridge, 0) != changed
    monkeypatch.setattr(telemetry, "SOURCE_SHA256", "b" * 64)
    assert telemetry._build_key(build_fixture.binary, bridge, 1) != changed


def test_bad_cached_archive_rejected_without_header_extraction(build_fixture):
    build_fixture.cache.mkdir(parents=True)
    (build_fixture.cache / "source.tar.gz").write_bytes(b"not-verified")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        telemetry._library()
    assert not build_fixture.commands
    assert not list(build_fixture.cache.glob("headers-*"))


def test_bad_download_removed_before_retry(build_fixture, monkeypatch):
    monkeypatch.setattr(
        telemetry.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"bad")
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        telemetry._library()
    assert not (build_fixture.cache / "source.tar.gz").exists()
    assert not list(build_fixture.cache.glob("*.tmp*"))


def test_supplied_archive_is_verified_and_avoids_download(
    build_fixture, monkeypatch, tmp_path
):
    supplied = tmp_path / "provided-source.tar.gz"
    supplied.write_bytes(build_fixture.source)
    monkeypatch.setenv("KARTSTUDY_NATIVE_SOURCE_ARCHIVE", str(supplied))
    monkeypatch.setattr(
        telemetry.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("download"),
    )
    telemetry._library()
    assert (build_fixture.cache / "source.tar.gz").read_bytes() == supplied.read_bytes()


@pytest.mark.parametrize("failure", ["timeout", "compiler_error"])
def test_failed_compile_is_bounded_and_cleans_owned_temporaries(
    build_fixture, monkeypatch, failure
):
    def failed(command):
        Path(command[-1]).write_bytes(b"incomplete")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, telemetry.BUILD_TIMEOUT)
        return SimpleNamespace(returncode=1, stderr="compiler failed")

    monkeypatch.setattr(telemetry, "_run_compile", failed)
    with pytest.raises(RuntimeError, match="exceeded|build failed"):
        telemetry._library()
    assert not list(build_fixture.cache.glob("headers-*"))
    assert not list(build_fixture.cache.glob("telemetry-*"))


def test_missing_linux_compiler_is_explicit(build_fixture, monkeypatch):
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match=r"C\+\+17 compiler"):
        telemetry._library()
    assert not build_fixture.commands


def test_compiler_process_uses_own_group_and_bounded_communication(monkeypatch):
    calls = []
    process = SimpleNamespace(
        returncode=0,
        communicate=lambda **kwargs: calls.append(kwargs) or ("out", "err"),
    )

    def spawn(command, **kwargs):
        assert kwargs["start_new_session"] and kwargs["text"]
        assert kwargs["stdout"] == kwargs["stderr"] == subprocess.PIPE
        return process

    monkeypatch.setattr(telemetry.subprocess, "Popen", spawn)
    result = telemetry._run_compile(["c++"])
    assert (result.returncode, result.stdout, result.stderr) == (0, "out", "err")
    assert calls == [{"timeout": telemetry.BUILD_TIMEOUT}]


@pytest.mark.parametrize("interruption", ["timeout", "interrupt"])
def test_interrupted_compile_kills_and_reaps_owned_group(monkeypatch, interruption):
    calls, killed = [], []
    error = (
        subprocess.TimeoutExpired(["c++"], telemetry.BUILD_TIMEOUT)
        if interruption == "timeout"
        else KeyboardInterrupt()
    )

    def communicate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise error
        return "", ""

    monkeypatch.setattr(
        telemetry.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(pid=12345, communicate=communicate),
    )
    monkeypatch.setattr(
        telemetry.os, "killpg", lambda pid, sig: killed.append((pid, sig))
    )
    with pytest.raises(type(error)):
        telemetry._run_compile(["c++"])
    assert killed == [(12345, signal.SIGKILL)]
    assert calls == [{"timeout": telemetry.BUILD_TIMEOUT}, {"timeout": 5}]


def test_compiler_abi_mismatch_rejected_before_timer_calls(build_fixture):
    build_fixture.native.kartstudy_telemetry_cpp_string_abi.value = 0
    with pytest.raises(RuntimeError, match="does not match"):
        telemetry._library()


def test_unqualified_linux_binary_fails_before_cache_or_build(build_fixture):
    build_fixture.binary.write_bytes(b"unqualified-simulator")
    with pytest.raises(RuntimeError, match="binary fingerprint"):
        telemetry._library()
    assert not build_fixture.cache.exists()
    assert not build_fixture.commands


@pytest.mark.parametrize(
    "machine,implementation", [("aarch64", "CPython"), ("x86_64", "PyPy")]
)
def test_unqualified_linux_runtime_fails_closed(
    build_fixture, monkeypatch, machine, implementation
):
    monkeypatch.setattr(telemetry.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        telemetry.platform, "python_implementation", lambda: implementation
    )
    with pytest.raises(RuntimeError, match="CPython 3.12 on x86_64"):
        telemetry._library()
    assert not build_fixture.cache.exists()


def test_lock_wait_has_deadline(monkeypatch):
    monkeypatch.setattr(telemetry.time, "monotonic", iter((0, 60)).__next__)
    monkeypatch.setattr(telemetry.time, "sleep", lambda duration: pytest.fail("sleep"))

    def busy(*args):
        raise BlockingIOError

    monkeypatch.setattr(telemetry.fcntl, "flock", busy)
    with pytest.raises(RuntimeError, match="build lock"):
        telemetry._lock_build(object())


@pytest.mark.parametrize("timer", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_native_read_is_error_not_no_rescue(monkeypatch, timer):
    monkeypatch.setattr(
        telemetry,
        "_library",
        lambda: SimpleNamespace(kartstudy_rescue_time_left=Function(timer)),
    )
    with pytest.raises(RuntimeError, match="nonfinite"):
        telemetry.rescue_time_left()


@pytest.mark.parametrize("timer,expected", [(-1.0, None), (0.0, 0.0), (1.5, 1.5)])
def test_native_timer_retains_rescue_and_no_rescue_semantics(
    monkeypatch, timer, expected
):
    monkeypatch.setattr(
        telemetry,
        "_library",
        lambda: SimpleNamespace(kartstudy_rescue_time_left=Function(timer)),
    )
    assert telemetry.rescue_time_left() == expected


@pytest.mark.skipif(
    os.environ.get("KARTSTUDY_RUN_NATIVE_TELEMETRY_TESTS") != "1",
    reason="Real simulator telemetry needs the coordinator's admitted native lane",
)
def test_real_rescue_activation_readonly_clock_and_reset(monkeypatch):
    from kartstudy import environment

    telemetry._library.cache_clear()
    assert telemetry.rescue_time_left() is None  # No race initialized in this worker.
    assert telemetry.rescue_time_left(-1) is None
    step = 0
    original_decode = environment.decode_action

    def rescue_at_boundary(action):
        physical = original_decode(action)
        physical["rescue"] = int(step == 60)
        return physical

    monkeypatch.setattr(environment, "decode_action", rescue_at_boundary)
    env = environment.DrivingEnv()
    try:
        env.reset(seed=0)
        world = env.env.unwrapped.world
        before = (float(world.time), len(world.karts))
        for _ in range(10):
            assert telemetry.rescue_time_left() is None
        assert (float(world.time), len(world.karts)) == before
        active = []
        for step in range(100):
            _, _, _, _, info = env.step(env.expert_action())
            if info["rescue_active"]:
                active.append(telemetry.rescue_time_left())
        assert len(active) > 1 and all(
            timer is not None and timer >= 0 for timer in active
        )
        assert info["rescue_count"] == 1
        env.reset(seed=1)
        assert telemetry.rescue_time_left() is None
        assert env.driving_metrics.snapshot()["rescue_count"] == 0
    finally:
        env.close()
    assert telemetry.rescue_time_left() is None
