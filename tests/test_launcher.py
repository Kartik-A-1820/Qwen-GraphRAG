import sys

import pytest

import run_project


class Process:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode


def test_busy_port_does_not_start_services(monkeypatch):
    monkeypatch.setattr(run_project, "port_available", lambda port: False)
    monkeypatch.setattr(run_project.subprocess, "Popen", lambda *a, **kw: pytest.fail("unexpected child process"))
    with pytest.raises(SystemExit) as error:
        run_project.main([])
    assert error.value.code == 2


def test_failed_backend_is_detected_without_waiting_for_timeout():
    with pytest.raises(RuntimeError, match="status 7"):
        run_project.wait_for_backend(Process(returncode=7), 8000, timeout=300)


def test_frontend_launch_failure_stops_backend(monkeypatch):
    backend = Process()
    monkeypatch.setattr(run_project, "port_available", lambda port: True)
    monkeypatch.setattr(run_project, "wait_for_backend", lambda *a: None)
    calls = []

    def spawn(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return backend
        raise OSError("frontend unavailable")

    monkeypatch.setattr(run_project.subprocess, "Popen", spawn)
    assert run_project.main([]) == 1
    assert backend.terminated


def test_active_interpreter_custom_api_url_and_cleanup(monkeypatch):
    backend, frontend = Process(), Process(returncode=0)
    monkeypatch.setattr(run_project, "port_available", lambda port: True)
    monkeypatch.setattr(run_project, "wait_for_backend", lambda *a: None)
    monkeypatch.delenv("RAG_BACKEND_MODULE", raising=False)
    calls = []

    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        return backend if len(calls) == 1 else frontend

    monkeypatch.setattr(run_project.subprocess, "Popen", spawn)
    assert run_project.main(["--api-port", "8100", "--ui-port", "8600"]) == 0
    assert calls[0][0][:4] == [sys.executable, "-m", "uvicorn", "backend.main:app"]
    assert "127.0.0.1" in calls[0][0]
    assert calls[1][1]["env"]["RAG_API_URL"] == "http://127.0.0.1:8100"
    assert calls[1][0][0] == sys.executable
    assert "8600" in calls[1][0]
    assert backend.terminated
