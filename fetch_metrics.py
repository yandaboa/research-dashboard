"""Poll wandb for the metric series linked to ledger experiments and cache them under <root>/metrics/.

Run standalone (cron, every 5 min) — no agents involved:
    python3 fetch_metrics.py --sweep
Ad-hoc:
    python3 fetch_metrics.py --run-url https://wandb.ai/<entity>/<project>/runs/<id> --metric <key>
"""

from __future__ import annotations

import argparse
import base64
import json
import netrc
import os
import re
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import unquote, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ledger import atomic_write, ledger_root, load_dir_entries, metrics_dir, now_ts, subdirs  # noqa: E402

WANDB_HOST = "api.wandb.ai"
GRAPHQL_URL = "https://api.wandb.ai/graphql"
HISTORY_QUERY = (
    "query($e:String!,$p:String!,$r:String!,$s:[JSONString!]!)"
    "{ project(entityName:$e,name:$p){ run(name:$r){ state sampledHistory(specs:$s) } } }"
)
DEFAULT_SAMPLES = 400
DEAD_EXP_STATUSES = ("done", "killed", "failed")
DEAD_RUN_STATES = ("finished", "crashed")
RUN_PATH_RE = re.compile(r"^/([^/]+)/([^/]+)/runs/([^/?#]+)")


def parse_run_url(url: str) -> tuple[str, str, str]:
    """https://wandb.ai/<entity>/<project>/runs/<run_id> -> (entity, project, run_id)."""
    assert isinstance(url, str) and url.strip(), "run_url must be a non-empty string"
    parsed = urlparse(url.strip())
    assert parsed.scheme in ("http", "https"), f"run_url must be an http(s) url: {url}"
    match = RUN_PATH_RE.match(parsed.path)
    assert match, f"run_url must be a RUN url (.../<entity>/<project>/runs/<run_id>), got: {url}"
    entity, project, run_id = (unquote(g) for g in match.groups())
    return entity, project, run_id


def api_key() -> str:
    path = os.path.expanduser("~/.netrc")
    assert os.path.exists(path), f"~/.netrc not found; wandb auth needs a '{WANDB_HOST}' entry (run `wandb login`)"
    auth = netrc.netrc(path).authenticators(WANDB_HOST)
    assert auth is not None, f"no '{WANDB_HOST}' entry in ~/.netrc; run `wandb login`"
    assert auth[2], f"empty api key for '{WANDB_HOST}' in ~/.netrc; run `wandb login`"
    return auth[2]


def graphql(query: str, variables: dict, timeout: int = 30, attempts: int = 3) -> dict:
    body = json.dumps({"query": query, "variables": variables}).encode()
    token = base64.b64encode(f"api:{api_key()}".encode()).decode()
    headers = {"Content-Type": "application/json", "Authorization": f"Basic {token}"}
    request = urllib.request.Request(GRAPHQL_URL, data=body, headers=headers)
    payload = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
            break
        except urllib.error.HTTPError as exc:  # 4xx/5xx: no point retrying a rejected query
            detail = exc.read().decode(errors="replace")[:200]
            raise RuntimeError(f"wandb graphql HTTP {exc.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"wandb graphql request failed: {exc.__class__.__name__}: {exc}") from None
            time.sleep(2 * (attempt + 1))
    errors = (payload or {}).get("errors")
    if errors:
        message = errors[0].get("message") if isinstance(errors[0], dict) else str(errors[0])
        raise RuntimeError(f"wandb graphql error: {message}")
    data = (payload or {}).get("data")
    assert isinstance(data, dict), "wandb graphql response has no data"
    return data


def fetch_series(run_url: str, key: str, samples: int = DEFAULT_SAMPLES) -> dict:
    """{run_state, points: [[step, value], ...], latest} for one (run, metric key)."""
    assert key and key.strip(), "metric key must not be empty"
    entity, project, run_id = parse_run_url(run_url)
    spec = json.dumps({"keys": ["_step", key], "samples": int(samples)})
    data = graphql(HISTORY_QUERY, {"e": entity, "p": project, "r": run_id, "s": [spec]})
    project_data = data.get("project")
    assert project_data, f"unknown wandb project: {entity}/{project}"
    run = project_data.get("run")
    assert run, f"unknown wandb run: {entity}/{project}/{run_id}"

    history = run.get("sampledHistory") or []
    rows = history[0] if history else []
    if isinstance(rows, str):  # some deployments return the series as a JSON string
        rows = json.loads(rows)
    points = []
    for row in rows:
        if not isinstance(row, dict) or key not in row:
            continue
        value = row[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        points.append([row.get("_step"), float(value)])
    points = [p for p in points if isinstance(p[0], (int, float))]
    points.sort(key=lambda p: p[0])
    assert points, f"no logged points for metric key {key!r} in {entity}/{project}/{run_id} (check the key name)"
    return {"run_state": run.get("state") or "unknown", "points": points, "latest": points[-1][1]}


def normalize_spec(spec: dict) -> dict:
    """Ledger metric spec -> {run_url, key, label} with the default label filled in."""
    assert isinstance(spec, dict), "each metric must be an object with run_url and key"
    run_url = str(spec.get("run_url") or "").strip()
    key = str(spec.get("key") or "").strip()
    parse_run_url(run_url)
    assert key, "metric key must not be empty"
    label = str(spec.get("label") or "").strip() or key.rsplit("/", 1)[-1]
    return {"run_url": run_url, "key": key, "label": label}


def entry_metrics(entry: dict) -> list[dict]:
    return [m for m in (entry.get("metrics") or []) if isinstance(m, dict)]


def metrics_path(root: str, exp_id: str) -> str:
    return os.path.join(metrics_dir(root), f"{exp_id}.json")


def load_cache(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def can_skip(entry: dict, cache: dict | None, specs: list[dict]) -> bool:
    """Dead run + a complete cache of dead series never changes again."""
    if entry.get("status") not in DEAD_EXP_STATUSES or not cache:
        return False
    cached = cache.get("series") or []
    if not cached:
        return False
    have = {(s.get("run_url"), s.get("key")) for s in cached if isinstance(s, dict)}
    if any((spec["run_url"], spec["key"]) not in have for spec in specs):
        return False  # a metric was added after the last sweep
    return all(s.get("error") is None and s.get("run_state") in DEAD_RUN_STATES for s in cached)


def sweep_experiment(root: str, entry: dict, samples: int, force: bool) -> str:
    exp_id = entry.get("id") or ""
    specs = [normalize_spec(spec) for spec in entry_metrics(entry)]
    path = metrics_path(root, exp_id)
    cache = load_cache(path)
    if not force and can_skip(entry, cache, specs):
        return f"{exp_id}: skipped (status {entry.get('status')}, cached series all finished)"

    series, n_err = [], 0
    for spec in specs:
        item = dict(spec, run_state="unknown", points=[], latest=None, error=None)
        try:
            item.update(fetch_series(spec["run_url"], spec["key"], samples))
        except (AssertionError, RuntimeError) as exc:
            item["error"] = str(exc) or exc.__class__.__name__
            n_err += 1
        series.append(item)

    payload = {"exp_id": exp_id, "fetched": now_ts(), "series": series}
    os.makedirs(metrics_dir(root), exist_ok=True)
    atomic_write(path, json.dumps(payload, indent=2) + "\n")
    return f"{exp_id}: {len(series) - n_err} ok, {n_err} err"


def cmd_sweep(args: argparse.Namespace) -> None:
    root = args.root or ledger_root()
    assert os.path.isdir(root), f"ledger root does not exist: {root}"
    _, _, exp_dir, _ = subdirs(root)
    for entry in load_dir_entries(exp_dir):
        if not entry_metrics(entry):
            continue
        try:
            print(sweep_experiment(root, entry, args.samples, args.force), flush=True)
        except Exception as exc:  # one bad experiment must not end the sweep
            print(f"{entry.get('id')}: FAILED {exc.__class__.__name__}: {exc}", file=sys.stderr, flush=True)


def cmd_direct(args: argparse.Namespace) -> None:
    series = fetch_series(args.run_url, args.metric, args.samples)
    print(json.dumps({"run_url": args.run_url, "key": args.metric, **series}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch wandb metric series for ledger experiments.")
    parser.add_argument("--sweep", action="store_true", help="refresh the metric cache of every experiment")
    parser.add_argument("--root", default=None, help="ledger root (default: UWLAB_LEDGER_ROOT)")
    parser.add_argument("--force", action="store_true", help="sweep: refetch even finished runs")
    parser.add_argument("--run-url", help="direct mode: https://wandb.ai/<entity>/<project>/runs/<id>")
    parser.add_argument("--metric", help="direct mode: wandb metric key")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.sweep:
            assert not args.run_url and not args.metric, "--sweep takes no --run-url/--metric"
            cmd_sweep(args)
            return
        assert args.run_url and args.metric, "use --sweep, or --run-url URL --metric KEY for a direct fetch"
        cmd_direct(args)
    except (AssertionError, RuntimeError) as exc:
        print(f"error: {exc or exc.__class__.__name__}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
