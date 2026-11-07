import json
import os
import runpy
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kartstudy import train


@pytest.fixture
def wrapper(tmp_path, monkeypatch):
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/wsl_run.py")
    )
    namespace = module["wsl_environment"].__globals__
    library, drivers = tmp_path / "lib", tmp_path / "drivers"
    active = drivers / "active-nvidia-driver"
    library.mkdir()
    active.mkdir(parents=True)
    (library / "libcuda.so.1").touch()
    (active / "libnvidia-ptxjitcompiler.so.1").touch()
    monkeypatch.setitem(namespace, "WSL_LIB", library)
    monkeypatch.setitem(namespace, "DRIVER_ROOT", drivers)
    monkeypatch.setattr(module["platform"], "system", lambda: "Linux")
    monkeypatch.setattr(
        module["platform"], "release", lambda: "6.6-microsoft-standard-WSL2"
    )
    monkeypatch.setattr(
        module["subprocess"],
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout=json.dumps({"driver_directory": str(active)})
        ),
    )
    return module, library, active


def test_wrapper_uses_mapped_driver_and_preserves_other_library_paths(
    wrapper, monkeypatch
):
    module, library, active = wrapper
    monkeypatch.setenv("LD_LIBRARY_PATH", f"/old-nvidia:{library}:/other")
    environment, directory = module["wsl_environment"]()
    assert directory == active
    assert environment["LD_LIBRARY_PATH"] == f"{library}:{active}:/old-nvidia:/other"
    assert module["os"].environ["LD_LIBRARY_PATH"] == f"/old-nvidia:{library}:/other"


def test_wrapper_pins_current_fusion_default_and_honors_explicit_choice(
    wrapper, monkeypatch
):
    module, _, _ = wrapper
    monkeypatch.delenv("TRITON_DEFAULT_FP_FUSION", raising=False)
    assert module["wsl_environment"]()[0]["TRITON_DEFAULT_FP_FUSION"] == "1"
    monkeypatch.setenv("TRITON_DEFAULT_FP_FUSION", "0")
    assert module["wsl_environment"]()[0]["TRITON_DEFAULT_FP_FUSION"] == "0"


def test_wrapper_refuses_missing_matching_ptx_compiler(wrapper):
    module, _, active = wrapper
    (active / "libnvidia-ptxjitcompiler.so.1").unlink()
    with pytest.raises(RuntimeError, match="Matching WSL PTX compiler"):
        module["wsl_environment"]()


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(-11, "probe", stderr="driver crash"),
        subprocess.TimeoutExpired("probe", 15),
    ],
)
def test_wrapper_bounds_and_reports_driver_probe_failure(wrapper, monkeypatch, failure):
    module, _, _ = wrapper

    def fail(*a, **kw):
        assert kw["timeout"] == 15
        raise failure

    monkeypatch.setattr(module["subprocess"], "run", fail)
    with pytest.raises(RuntimeError, match="WSL CUDA driver probe failed"):
        module["wsl_environment"]()


def test_cuda_requested_but_unavailable_does_not_create_a_run(tmp_path, monkeypatch):
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    output = tmp_path / "refused"
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        train.run(train.defaults() | {"device": "cuda"}, output)
    assert not output.exists()


@pytest.mark.parametrize("unresponsive", [False, True])
def test_readiness_timeout_interrupts_then_bounds_hard_shutdown(
    tmp_path, monkeypatch, unresponsive
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/check_wsl.py")
    )
    calls, signals = [], []

    def communicate(timeout):
        calls.append(timeout)
        if len(calls) == 1 or (unresponsive and len(calls) == 2):
            raise subprocess.TimeoutExpired("worker", timeout)
        return "partial output", ""

    process = SimpleNamespace(pid=123, returncode=-2, communicate=communicate)
    monkeypatch.setattr(module["subprocess"], "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(
        module["os"], "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    with pytest.raises(RuntimeError, match="Readiness check timed out"):
        module["checked"](["worker"], {}, tmp_path / "output.log", 180)
    assert calls == ([180, 5, 5] if unresponsive else [180, 5])
    assert signals == (
        [(123, signal.SIGINT), (123, signal.SIGKILL)]
        if unresponsive
        else [(123, signal.SIGINT)]
    )
    assert "partial output" in (tmp_path / "output.log").read_text()


def test_readiness_timeout_reaps_separately_owned_compiler(tmp_path, monkeypatch):
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/check_wsl.py")
    )
    child = "import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    parent = "import sys; from kartstudy.native_telemetry import _run_compile; _run_compile([sys.executable, '-c', sys.argv[1], sys.argv[2]])"
    pidfile = tmp_path / "compiler.pid"
    with pytest.raises(RuntimeError, match="Readiness check timed out"):
        module["checked"](
            [sys.executable, "-c", parent, child, str(pidfile)],
            os.environ.copy(),
            tmp_path / "output.log",
            4,
        )
    assert pidfile.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_readiness_source_failure_cannot_leave_a_passed_report(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/check_wsl.py")
    )
    namespace = module["main"].__globals__
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "driver"
    metadata = {
        "libraries": {
            str(directory / "libcuda.so.1"): "driver",
            str(directory / "libnvidia-ptxjitcompiler.so.1"): "compiler",
        }
    }
    monkeypatch.setitem(namespace, "wsl_environment", lambda: ({}, directory))

    def checked(command, environment, output, timeout):
        if timeout == 30:
            return json.dumps(metadata)
        cases = (
            '<testcase name="test_real_rescue_activation_readonly_clock_and_reset"/>'
        )
        cases += "".join(f'<testcase name="cuda-fixture-{i}"/>' for i in range(5))
        (output.parent / "fixtures.xml").write_text(f"<testsuite>{cases}</testsuite>")
        return "passed"

    monkeypatch.setitem(namespace, "checked", checked)
    for name in (
        "src/kartstudy/native_telemetry.cpp",
        "tests/test_snapshots.py",
        "tests/test_native_telemetry.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    # All runtime checks pass, but the required lock is absent during hashing.
    output = tmp_path / "report"
    with pytest.raises(SystemExit) as error:
        module["main"](["--output", str(output)])
    assert error.value.code == 1
    report = json.loads((output / "readiness.json").read_text())
    assert report["status"] == "failed" and "uv.lock" in report["error"]
