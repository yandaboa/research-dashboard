"""Local web viewer for the experiment/diff ledger written by scripts/tools/ledger.py."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ledger import atomic_write, ledger_root, metrics_dir, now_ts, statuses_for, subdirs  # noqa: E402

VIEWER_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger_viewer.html")
EDITABLE_FIELDS = ("status", "notes", "results", "conclusion", "summary", "title", "takeaways", "text", "why")
LIST_FIELDS = ("takeaways",)  # edited as one-per-line text in the GUI, posted as a list


def load_dir(path: str) -> list[dict]:
    entries = []
    if not os.path.isdir(path):
        return entries
    for name in sorted(os.listdir(path)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(path, name)) as f:
                entries.append(json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping {name}: {exc}", flush=True)
    return entries


def find_entry(root: str, entry_id: str) -> tuple[str, dict] | tuple[None, None]:
    for d in subdirs(root)[:4]:
        path = os.path.join(d, f"{entry_id}.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return path, json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"failed to read {path}: {exc}", flush=True)
                return None, None
    return None, None


class Handler(BaseHTTPRequestHandler):
    root = ledger_root()

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.log_date_time_string()} {self.address_string()} {fmt % args}", flush=True)

    def _send(self, code: int, body: bytes, content_type: str, no_store: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json; charset=utf-8", no_store=True)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            try:
                with open(VIEWER_HTML, "rb") as f:
                    body = f.read()
            except OSError as exc:
                self._send_json(500, {"error": f"cannot read viewer html: {exc}"})
                return
            self._send(200, body, "text/html; charset=utf-8", no_store=True)
            return

        if path == "/api/data":
            questions_dir, diffs_dir, exp_dir, truths_dir, _ = subdirs(self.root)
            payload = {
                "questions": load_dir(questions_dir),
                "diffs": load_dir(diffs_dir),
                "experiments": load_dir(exp_dir),
                "truths": load_dir(truths_dir),
                "generated": now_ts(),
            }
            self._send_json(200, payload)
            return

        if path.startswith("/api/metrics/"):
            entry_id = unquote(path[len("/api/metrics/") :])
            cache_path = os.path.join(metrics_dir(self.root), f"{os.path.basename(entry_id)}.json")
            if not entry_id or not os.path.exists(cache_path):
                self._send_json(404, {"error": "no metrics for this entry"})
                return
            with open(cache_path, "rb") as f:
                self._send(200, f.read(), "application/json; charset=utf-8", no_store=True)
            return

        if path.startswith("/api/patch/"):
            entry_id = unquote(path[len("/api/patch/") :])
            _, entry = find_entry(self.root, entry_id)
            if entry is None:
                self._send_json(404, {"error": "unknown id"})
                return
            rel = entry.get("patch")
            patch_path = os.path.join(self.root, rel) if rel else None
            if not patch_path or not os.path.exists(patch_path):
                self._send_json(404, {"error": "no patch for this entry"})
                return
            with open(patch_path, "rb") as f:
                self._send(200, f.read(), "text/plain; charset=utf-8", no_store=True)
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/update":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"bad json: {exc}"})
            return

        entry_id = body.get("id")
        fields = body.get("fields")
        if not isinstance(entry_id, str) or not isinstance(fields, dict) or not fields:
            self._send_json(400, {"error": "expected {id, fields}"})
            return

        entry_path, entry = find_entry(self.root, entry_id)
        if entry is None:
            self._send_json(404, {"error": "unknown id"})
            return

        kind = entry.get("kind", "diff")
        allowed_status = statuses_for(kind)
        for key, value in fields.items():
            if key not in EDITABLE_FIELDS:
                self._send_json(400, {"error": f"field not editable: {key}"})
                return
            if key in LIST_FIELDS:
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    self._send_json(400, {"error": f"field {key} must be a list of strings"})
                    return
                continue
            if not isinstance(value, str):
                self._send_json(400, {"error": f"field {key} must be a string"})
                return
            if key == "status" and value not in allowed_status:
                self._send_json(400, {"error": f"status for a {kind} must be one of {list(allowed_status)}"})
                return

        entry.update(fields)
        entry["updated"] = now_ts()
        atomic_write(entry_path, json.dumps(entry, indent=2) + "\n")
        self._send_json(200, entry)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the ledger viewer.")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--root", default=None)
    args = parser.parse_args()

    root = args.root or ledger_root()
    assert os.path.isdir(root), f"ledger root does not exist: {root} (run ledger.py add-diff first)"
    Handler.root = root

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ledger root: {root}")
    print(f"serving on http://{args.host}:{args.port}  ({datetime.now().isoformat(timespec='seconds')})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
