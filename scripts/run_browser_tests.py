#!/usr/bin/env python3
"""
Run the browser audio tests headlessly and report the result.

The DSP inside the AudioWorklet cannot be exercised by the Python suite —
it runs on a browser's audio render thread, against APIs that exist nowhere
else. `web/test.html` asserts it against signals whose correct answer is
known analytically; this drives that page and turns its outcome into an
exit code so CI can gate on it.

Chromium is launched with cross-origin isolation available, because the
capture path uses SharedArrayBuffer and the fallback is a different code
path — testing only the fallback would leave the one that actually runs
unverified.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import closing

TIMEOUT_S = 120


def free_port() -> int:
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for(url: str, timeout: float = 30.0) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.25)
    return False


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed; run `pip install playwright` "
              "and `playwright install chromium`", file=sys.stderr)
        return 2

    port = free_port()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"))
    server = subprocess.Popen(
        [sys.executable, "-m", "mixengine", "serve",
         "--port", str(port), "--data", os.path.join(root, "data")],
        cwd=root, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        base = f"http://127.0.0.1:{port}"
        if not wait_for(f"{base}/api/health"):
            print("the server never became healthy", file=sys.stderr)
            return 1

        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=[
                # Without a real output device the audio graph never renders,
                # and every worklet assertion times out.
                "--autoplay-policy=no-user-gesture-required",
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
            ])
            page = browser.new_page()
            failures: list[str] = []
            page.on("pageerror", lambda e: failures.append(str(e)))
            page.goto(f"{base}/static/test.html")
            page.wait_for_function(
                "document.title.startsWith('✓') || document.title.startsWith('✗')",
                timeout=TIMEOUT_S * 1000)

            summary = page.text_content("#summary") or ""
            rows = page.eval_on_selector_all(
                ".t.fail", "els => els.map(e => e.textContent.trim())")
            browser.close()

        print(summary)
        for r in rows:
            print(f"  FAIL  {r}", file=sys.stderr)
        for f in failures:
            print(f"  ERROR {f}", file=sys.stderr)
        return 0 if not rows and not failures else 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    sys.exit(main())
