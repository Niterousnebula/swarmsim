"""Local web server for the control app (standard library only).

    GET  /                      the page
    GET  /api/stream            live simulation state (server-sent events)
    GET  /api/scene             what is loaded: meshes, textures, vehicle models
    GET  /api/asset/<load>/<id> a point cloud of the loaded city, or a vehicle mesh (vertex buffer)
    GET  /api/models            everything that can be flown
    GET  /api/cities            cities to choose from and the limits of the scenario settings
    GET  /api/train             training progress
    POST /api/cmd               {"type": "probe", ...}    what a click would place (live preview)
                                {"type": "load" | "setup" | "generate" | "place" | "unplace"      (answered when done)
                                         | "pause" | "speed" | "reset" | "pilot" | "keys"
                                         | "train_start" | "train_stop", ...}
"""
from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .cloud import city_cloud
from .meshes import vertex_buffer
from .scenario import HEIGHT_RANGE, LIMITS, MAX_DRONES, PATTERNS, SIZES, STYLES, list_cities
from .sim import Sim
from .trainer import Trainer, list_models

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
SYNC = {"load", "setup", "generate", "place", "unplace"}    # commands whose result the page shows
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".png": "image/png", ".svg": "image/svg+xml"}


def _handler(sim: Sim, trainer: Trainer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):             # keep the terminal quiet
            pass

        # ---------------------------------------------------------------- replies
        def _send(self, body: bytes, ctype: str, code: int = 200, cache: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(json.dumps(obj).encode(), "application/json", code)

        # -------------------------------------------------------------------- GET
        def do_GET(self):
            path = self.path.split("?")[0]
            try:
                if path == "/":
                    path = "/index.html"
                if path == "/api/stream":
                    return self._stream()
                if path == "/api/scene":
                    return self._json(sim.scene or {})
                if path == "/api/models":
                    return self._json(list_models())
                if path == "/api/cities":
                    return self._json({"cities": list_cities(), "limits": LIMITS, "max_drones": MAX_DRONES,
                                       "styles": {k: v["label"] for k, v in STYLES.items()},
                                       "patterns": PATTERNS, "sizes": SIZES, "heights": HEIGHT_RANGE})
                if path == "/api/train":
                    return self._json(trainer.status())
                if path.startswith("/api/asset/"):
                    return self._asset(path)
                file = (STATIC / path.lstrip("/")).resolve()
                if STATIC in file.parents and file.is_file():
                    return self._send(file.read_bytes(), TYPES.get(file.suffix, "application/octet-stream"))
                self._json({"error": "not found"}, 404)
            except (BrokenPipeError, ConnectionError):
                pass

        def _asset(self, path: str):
            load, _, key = path[len("/api/asset/"):].partition("/")
            entry = sim.assets.get(key) if sim.scene and load == sim.asset_tag() else None
            if entry is None:
                return self._json({"error": "not part of the loaded scene"}, 404)
            file, scale = entry
            if isinstance(file, tuple):                         # ("cloud", city folder, kind)
                return self._send(city_cloud(file[1])[file[2]]["data"], "application/octet-stream", cache=True)
            if file.suffix.lower() == ".png":
                return self._send(file.read_bytes(), "image/png", cache=True)
            self._send(vertex_buffer(file, scale), "application/octet-stream", cache=True)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")       # the body ends when either side hangs up
            self.end_headers()
            self.close_connection = True
            last = -1
            while True:
                with sim.cond:
                    sim.cond.wait_for(lambda: sim.seq != last, timeout=1.0)
                    last, data = sim.seq, sim.snapshot
                self.wfile.write(b"data: " + data.encode() + b"\n\n")
                self.wfile.flush()

        # ------------------------------------------------------------------- POST
        def do_POST(self):
            try:
                if self.path != "/api/cmd":
                    return self._json({"error": "not found"}, 404)
                host = (self.headers.get("Host") or "").split(":")[0]
                if host not in ("127.0.0.1", "localhost"):          # only this machine's own page may send commands
                    return self._json({"error": "forbidden"}, 403)
                cmd = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                kind = cmd.get("type")
                if kind == "probe":                                 # live placement preview, answered right here
                    return self._json(sim.probe(cmd))
                if kind in SYNC:                                    # wait for the simulator's answer
                    out = sim.call(cmd)
                    return self._json(out, 400 if "error" in out else 200)
                if kind == "train_start":
                    out = trainer.start(cmd.get("name", ""), cmd.get("timesteps", 0),
                                        bool(cmd.get("defence")), cmd.get("init") or None)
                elif kind == "train_stop":
                    out = trainer.stop()
                else:
                    sim.send(cmd)
                    out = {"ok": True}
                self._json(out)
            except ValueError as exc:                               # a message for the person at the page
                self._json({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionError):
                pass

    return Handler


def serve(model: str = "stage1_navigate", port: int = 8765, open_browser: bool = True):
    """Start the simulation and the page, and block until Ctrl+C."""
    os.chdir(ROOT)                                # configs, cache/ and runs/ are relative to the project
    sim, trainer = Sim(model), Trainer()
    ThreadingHTTPServer.daemon_threads = True
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), _handler(sim, trainer))
    except OSError as exc:
        raise SystemExit(f"Port {port} is in use ({exc}). Is the app already running? "
                         f"Otherwise pick another one with --port.")
    sim.start()
    url = f"http://127.0.0.1:{port}/"
    print(f"swarmsim is running at {url}  (Ctrl+C to quit)")
    if open_browser:
        threading.Timer(0.6, webbrowser.open, [url]).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        sim.stopping = True
        trainer.shutdown()
        httpd.server_close()
