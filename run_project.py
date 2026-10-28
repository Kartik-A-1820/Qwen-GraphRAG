"""Launch the API and dashboard using the active, already prepared environment."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parent


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def wait_for_backend(process, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Backend exited with status {process.returncode} before becoming ready.")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if json.load(response).get("status") == "ok":
                    return
        except (OSError, URLError, ValueError):
            pass
        time.sleep(0.25)
    raise TimeoutError("Backend startup timed out. Inspect its logs or increase --startup-timeout.")


def stop_process(process) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Launch Qwen-GraphRAG from the active Python environment.")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument("--startup-timeout", type=float, default=300)
    args = parser.parse_args(argv)
    if not 0 < args.api_port < 65536 or not 0 < args.ui_port < 65536:
        parser.error("Ports must be between 1 and 65535.")
    if args.api_port == args.ui_port or args.startup_timeout <= 0:
        parser.error("Use distinct ports and a positive startup timeout.")
    for port in (args.api_port, args.ui_port):
        if not port_available(port):
            parser.error(f"Port {port} is already in use.")

    processes = []
    try:
        backend = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            os.getenv("RAG_BACKEND_MODULE", "backend.main:app"),
            "--host", "127.0.0.1", "--port", str(args.api_port),
        ], cwd=PROJECT_ROOT)
        processes.append(backend)
        wait_for_backend(backend, args.api_port, args.startup_timeout)
        frontend_env = os.environ.copy()
        frontend_env["RAG_API_URL"] = f"http://127.0.0.1:{args.api_port}"
        frontend = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "frontend" / "app.py"),
            "--server.address", "127.0.0.1", "--server.port", str(args.ui_port),
        ], cwd=PROJECT_ROOT, env=frontend_env)
        processes.append(frontend)
        print(f"Qwen-GraphRAG: http://127.0.0.1:{args.ui_port} (Ctrl+C stops both services)")
        while all(process.poll() is None for process in processes):
            time.sleep(0.25)
        return next((process.returncode for process in processes if process.returncode), 0)
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f"Qwen-GraphRAG startup failed: {exc}", file=sys.stderr)
        return 1
    finally:
        for process in reversed(processes):
            stop_process(process)


if __name__ == "__main__":
    raise SystemExit(main())
