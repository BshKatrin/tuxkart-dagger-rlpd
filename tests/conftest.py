import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_simulator_ipc(tmp_path_factory):
    # Headless PySTK's rich-presence callback can dereference a missing kart model.
    with pytest.MonkeyPatch.context() as monkeypatch:
        if sys.platform == "darwin":
            runtime = tmp_path_factory.mktemp("simulator-ipc")
            monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
        yield
