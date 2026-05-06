"""
Entry point for the PyTheraTwin web UI server.

Checks and installs required web dependencies, stops an older PyTheraTwin web UI
server on the same port when one is found, then launches the FastAPI
application via uvicorn.  An
OS browser window is opened automatically unless --no-browser is passed.

Usage
-----
    python run_server.py [--host HOST] [--port PORT] [--no-browser]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Ensure repo root is on the Python path so `web` package resolves correctly
# regardless of the directory from which this script is invoked.
REPO_ROOT = Path(__file__).parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _pid_file_path(port: int) -> Path:
    """Return the pid-file path for one bound web-server port."""
    return REPO_ROOT / f".pytheratwin_webui_server_{port}.json"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the server launcher."""
    p = argparse.ArgumentParser(description="Launch the PyTheraTwin web UI server.")
    p.add_argument("--port", type=int, default=8766, help="Port to listen on (default: 8766).")
    p.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1).")
    p.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser.")
    return p.parse_args()


def _process_command(pid: int) -> str:
    """Return the process command line for *pid*, or an empty string on failure."""
    try:
        result = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _looks_like_pytheratwin_server(pid: int) -> bool:
    """Return True when *pid* looks like this repo's web UI server."""
    cmd = _process_command(pid)
    if not cmd:
        return False
    return (
        "run_server.py" in cmd
        or "web.server:app" in cmd
        or (str(REPO_ROOT) in cmd and "uvicorn" in cmd)
    )


def _read_pid_file(port: int) -> dict | None:
    """Read the persisted PyTheraTwin server pid file if it exists and is valid JSON."""
    pid_file = _pid_file_path(port)
    if not pid_file.exists():
        return None
    try:
        return json.loads(pid_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_pid_file(host: str, port: int) -> None:
    """Persist the current PyTheraTwin web server process metadata."""
    payload = {
        "pid": os.getpid(),
        "host": host,
        "port": port,
        "repo_root": str(REPO_ROOT),
        "started_at": time.time(),
    }
    _pid_file_path(port).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _remove_pid_file(port: int) -> None:
    """Best-effort cleanup of the persisted pid file."""
    try:
        _pid_file_path(port).unlink(missing_ok=True)
    except Exception:
        pass


def _is_pid_alive(pid: int) -> bool:
    """Return True when a process with *pid* still exists."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _stop_pid(pid: int) -> None:
    """Terminate one process, escalating to SIGKILL only if needed."""
    if not _is_pid_alive(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not _is_pid_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    time.sleep(0.2)


def _listener_pids(port: int) -> list[int]:
    """Return listener PIDs on *port* when lsof is available, else an empty list."""
    try:
        result = subprocess.run(
            ["lsof", "-tiTCP:%d" % port, "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
        )
        return [int(pid) for pid in result.stdout.split() if pid.strip().isdigit()]
    except Exception:
        return []


def _port_in_use(host: str, port: int) -> bool:
    """Return True when binding to *(host, port)* would fail."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
        return False
    except OSError:
        return True
    finally:
        sock.close()


def _maybe_stop_existing_pytheratwin_server(host: str, port: int) -> None:
    """
    Stop only a prior PyTheraTwin web UI server on *port*.

    Never kills an unrelated process. If the port remains busy afterwards, the
    caller should fail with a clear message and let the user decide what to do.
    """
    pid_info = _read_pid_file(port)
    if pid_info:
        pid = int(pid_info.get("pid", 0) or 0)
        if pid and pid != os.getpid():
            if _looks_like_pytheratwin_server(pid) and int(pid_info.get("port", -1)) == port:
                _stop_pid(pid)
            elif not _is_pid_alive(pid):
                _remove_pid_file(port)

    for pid in _listener_pids(port):
        if pid == os.getpid():
            continue
        if _looks_like_pytheratwin_server(pid):
            _stop_pid(pid)

    if _port_in_use(host, port):
        raise RuntimeError(
            f"Port {port} is already in use by a non-PyTheraTwin process. "
            "The launcher will not kill unrelated services automatically."
        )


_WEB_DEPS = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.30",
    "python-multipart",
    "pillow",
    "nibabel",
    "pydicom",
]


def _ensure_deps() -> None:
    """Ensure web UI dependencies are importable.

    Checks importability directly — never parses pip output.
    If a key package is missing, installs all deps then re-execs once so
    the fresh process can import them.  If everything is already present,
    pip is never invoked at all.
    """
    print("Checking web UI dependencies…", flush=True)
    try:
        import uvicorn      # noqa: F401
        import fastapi      # noqa: F401
        import multipart    # noqa: F401
        import PIL          # noqa: F401
        import nibabel      # noqa: F401
        import pydicom      # noqa: F401
        print("Dependencies OK.", flush=True)
        return
    except ImportError:
        pass

    print("Installing missing dependencies…", flush=True)
    import subprocess as _sp
    result = _sp.run(
        [sys.executable, "-m", "pip", "install"] + _WEB_DEPS,
    )
    if result.returncode != 0:
        print("[ERROR] Failed to install dependencies. Run manually:")
        print(f"  pip install {' '.join(_WEB_DEPS)}")
        sys.exit(1)
    # Re-exec once so this process sees the newly installed packages.
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main() -> None:
    """Resolve dependencies, stop an older PyTheraTwin server if present, and start uvicorn."""
    args = _parse_args()
    url = f"http://{args.host}:{args.port}"

    print("=" * 60)
    print("  PyTheraTwin — Web UI")
    print("=" * 60)
    print(f"  URL : {url}")
    print("  Stop: Ctrl+C")
    print("=" * 60)
    print()

    _ensure_deps()
    try:
        _maybe_stop_existing_pytheratwin_server(args.host, args.port)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if not args.no_browser:
        def _open() -> None:
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn is not installed.")
        print("  Run:  pip install 'uvicorn[standard]' fastapi")
        sys.exit(1)

    pending_file = REPO_ROOT / ".pytheratwin_pending_run.json"

    try:
        _write_pid_file(args.host, args.port)
        uvicorn.run(
            "web.server:app",
            host=args.host,
            port=args.port,
            reload=False,
            log_level="warning",
        )
    except KeyboardInterrupt:
        pass
    finally:
        _remove_pid_file(args.port)

    # After the server exits, run any pipeline jobs queued by /api/run.
    if pending_file.exists():
        try:
            data = json.loads(pending_file.read_text(encoding="utf-8"))
            pending_file.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[ERROR] Could not read pending run file: {exc}")
            sys.exit(1)

        jobs = data.get("jobs", [])
        print(f"\nLaunching pipeline ({len(jobs)} patient(s))…\n")
        any_failed = False
        for job in jobs:
            result = subprocess.run(job["cmd"], cwd=str(REPO_ROOT))
            if result.returncode != 0:
                any_failed = True
        sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
