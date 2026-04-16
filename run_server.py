"""
Entry point for the Virtual Theranostic Trials web UI server.

Checks and installs required web dependencies, frees the target port if
already in use, then launches the FastAPI application via uvicorn.  An
OS browser window is opened automatically unless --no-browser is passed.

Usage
-----
    python run_server.py [--host HOST] [--port PORT] [--no-browser]
"""
from __future__ import annotations

import argparse
import os
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


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the server launcher."""
    p = argparse.ArgumentParser(description="Launch the VTT web UI server.")
    p.add_argument("--port", type=int, default=8766, help="Port to listen on (default: 8766).")
    p.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1).")
    p.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser.")
    return p.parse_args()


def _free_port(port: int) -> None:
    """Kill any process already listening on *port* so we can bind cleanly.

    Uses SIGTERM first (graceful), then SIGKILL after a short wait so that
    processes suspended with Ctrl+Z (SIGSTOP) are force-killed and the port
    is guaranteed to be released.
    """
    import signal
    import subprocess as _sp
    try:
        result = _sp.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True
        )
        pids = [p for p in result.stdout.strip().split() if p]
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(0.4)
        # Force-kill anything still alive (e.g. suspended via Ctrl+Z)
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass  
        time.sleep(0.2)   # give the OS a moment to release the port
    except Exception:
        pass   # lsof not available or nothing to kill — uvicorn will report the error


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
    """Resolve dependencies, free the port, and start the uvicorn server."""
    args = _parse_args()
    url = f"http://{args.host}:{args.port}"

    print("=" * 60)
    print("  Virtual Theranostic Trials — Web UI")
    print("=" * 60)
    print(f"  URL : {url}")
    print("  Stop: Ctrl+C")
    print("=" * 60)
    print()

    _ensure_deps()

    _free_port(args.port)

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

    try:
        uvicorn.run(
            "web.server:app",
            host=args.host,
            port=args.port,
            reload=False,
            log_level="warning",
        )
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
