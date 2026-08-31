"""Append/update entries in the central experiment + diff ledger shared by all worktrees."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DIFF_STATUSES = ("unreviewed", "questionable", "known_good")
EXP_STATUSES = ("planned", "running", "done", "killed", "failed")
QUESTION_STATUSES = ("open", "answered", "parked")


def ledger_root() -> str:
    return os.environ.get("UWLAB_LEDGER_ROOT", DEFAULT_ROOT)


def subdirs(root: str) -> tuple[str, str, str, str]:
    """(questions, diffs, experiments, patches) — entry dirs first, patches last."""
    return (
        os.path.join(root, "questions"),
        os.path.join(root, "diffs"),
        os.path.join(root, "experiments"),
        os.path.join(root, "patches"),
    )


def metrics_dir(root: str) -> str:
    """Cache of polled wandb metric series, one <exp-id>.json per experiment. Not an entry dir."""
    return os.path.join(root, "metrics")


def ensure_root(root: str) -> None:
    for d in (*subdirs(root), metrics_dir(root)):
        os.makedirs(d, exist_ok=True)


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60].strip("-") or "entry"


def split_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def git(args: list[str], cwd: str | None = None) -> tuple[int, str]:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip()


def detect_worktree(cwd: str | None = None) -> tuple[str, str, str]:
    """(repo, worktree, branch) from cwd git; ("unknown",)*3 outside a repo. worktree is "main" in the main checkout."""
    code, toplevel = git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if code != 0 or not toplevel:
        return "unknown", "unknown", "unknown"
    code, common = git(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=cwd)
    main_checkout = os.path.dirname(os.path.realpath(common)) if code == 0 and common else os.path.realpath(toplevel)
    repo = os.path.basename(main_checkout)
    worktree = "main" if os.path.realpath(toplevel) == main_checkout else os.path.basename(toplevel)
    code, branch = git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    return repo, worktree, (branch if code == 0 and branch else "unknown")


def capture_diff(mode: str, cwd: str | None = None) -> tuple[str, list[str]]:
    """(patch text, changed files) for the working tree or the branch range."""
    if mode == "working":
        diff_args = ["diff", "HEAD"]
        name_args = ["diff", "--name-only", "HEAD"]
    else:
        base = ""
        for trunk in ("main", "master"):
            code, base = git(["merge-base", trunk, "HEAD"], cwd=cwd)
            if code == 0 and base:
                break
        assert base, "could not find merge-base with main/master; use --capture working"
        diff_args = ["diff", f"{base}...HEAD"]
        name_args = ["diff", "--name-only", f"{base}...HEAD"]
    code, patch = git(diff_args, cwd=cwd)
    assert code == 0, f"git {' '.join(diff_args)} failed"
    code, names = git(name_args, cwd=cwd)
    files = names.splitlines() if code == 0 else []
    return patch, [f.strip() for f in files if f.strip()]


def entry_path(root: str, entry_id: str) -> str | None:
    for d in subdirs(root)[:3]:
        path = os.path.join(d, f"{entry_id}.json")
        if os.path.exists(path):
            return path
    return None


def unique_id(root: str, base_id: str) -> str:
    if entry_path(root, base_id) is None:
        return base_id
    n = 2
    while entry_path(root, f"{base_id}-{n}") is not None:
        n += 1
    return f"{base_id}-{n}"


def atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def write_entry(path: str, entry: dict) -> None:
    atomic_write(path, json.dumps(entry, indent=2) + "\n")


def load_entry(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def statuses_for(kind: str) -> tuple[str, ...]:
    if kind == "question":
        return QUESTION_STATUSES
    return DIFF_STATUSES if kind == "diff" else EXP_STATUSES


def assert_question_exists(root: str, question_id: str) -> None:
    questions_dir, _, _, _ = subdirs(root)
    assert os.path.exists(os.path.join(questions_dir, f"{question_id}.json")), f"unknown question id: {question_id}"


def cmd_add_question(args: argparse.Namespace) -> None:
    root = ledger_root()
    ensure_root(root)
    assert args.status in QUESTION_STATUSES, f"status must be one of {QUESTION_STATUSES}"
    questions_dir, _, _, _ = subdirs(root)

    entry_id = unique_id(root, args.id or f"q-{slugify(args.title)}")
    ts = now_ts()
    entry = {
        "kind": "question",
        "id": entry_id,
        "title": args.title,
        "summary": args.summary or "",
        "status": args.status,
        "conclusion": "",
        "takeaways": [],
        "tags": split_list(args.tags),
        "assets": [],
        "created": ts,
        "updated": ts,
    }
    write_entry(os.path.join(questions_dir, f"{entry_id}.json"), entry)
    print(entry_id)


def cmd_add_diff(args: argparse.Namespace) -> None:
    root = ledger_root()
    ensure_root(root)
    assert args.status in DIFF_STATUSES, f"status must be one of {DIFF_STATUSES}"
    _, diffs_dir, _, _ = subdirs(root)
    if args.question:
        assert_question_exists(root, args.question)

    entry_id = args.id or f"{datetime.now().strftime('%Y-%m-%d')}-{slugify(args.title)}"
    entry_id = unique_id(root, entry_id)

    repo, worktree, branch = detect_worktree()
    patch_rel = None
    files = split_list(args.files)
    if args.capture:
        patch_text, changed = capture_diff(args.capture)
        if patch_text.strip():
            patch_rel = os.path.join("patches", f"{entry_id}.patch")
            atomic_write(os.path.join(root, patch_rel), patch_text + "\n")
            if not args.no_files:
                files = changed
        else:
            print(f"warning: --capture {args.capture} produced an empty diff; no patch stored", file=sys.stderr)

    ts = now_ts()
    entry = {
        "kind": "diff",
        "id": entry_id,
        "title": args.title,
        "summary": args.summary or "",
        "question_id": args.question or None,
        "repo": repo,
        "worktree": worktree,
        "branch": branch,
        "commits": split_list(args.commits),
        "files": files,
        "patch": patch_rel,
        "status": args.status,
        "notes": "",
        "tags": split_list(args.tags),
        "assets": [],
        "created": ts,
        "updated": ts,
    }
    write_entry(os.path.join(diffs_dir, f"{entry_id}.json"), entry)
    print(entry_id)


def cmd_add_exp(args: argparse.Namespace) -> None:
    root = ledger_root()
    ensure_root(root)
    assert args.status in EXP_STATUSES, f"status must be one of {EXP_STATUSES}"
    _, diffs_dir, exp_dir, _ = subdirs(root)
    if args.question:
        assert_question_exists(root, args.question)

    diff_ids = split_list(args.diffs)
    for diff_id in diff_ids:
        assert os.path.exists(os.path.join(diffs_dir, f"{diff_id}.json")), f"unknown diff id: {diff_id}"

    entry_id = args.id or f"exp-{datetime.now().strftime('%Y-%m-%d')}-{slugify(args.title)}"
    entry_id = unique_id(root, entry_id)

    ts = now_ts()
    entry = {
        "kind": "experiment",
        "id": entry_id,
        "title": args.title,
        "summary": args.summary or "",
        "question_id": args.question or None,
        "diff_ids": diff_ids,
        "status": args.status,
        "cluster": args.cluster or "",
        "job_ids": split_list(args.jobs),
        "wandb": split_list(args.wandb),
        "metrics": [],
        "results": "",
        "tags": split_list(args.tags),
        "assets": [],
        "created": ts,
        "updated": ts,
    }
    for spec in args.metric or []:
        add_metric(entry, parse_metric(spec))
    write_entry(os.path.join(exp_dir, f"{entry_id}.json"), entry)
    print(entry_id)


def append_dedup(entry: dict, key: str, values: list[str]) -> None:
    existing = list(entry.get(key) or [])
    for value in values:
        if value not in existing:
            existing.append(value)
    entry[key] = existing


def add_asset(entry: dict, label: str, location: str) -> None:
    """Append an {label, location} asset; dedup on location."""
    assert label.strip(), "asset label must not be empty"
    assert location.strip(), "asset location must not be empty"
    assets = [a for a in (entry.get("assets") or []) if isinstance(a, dict)]
    if any(a.get("location") == location.strip() for a in assets):
        entry["assets"] = assets
        return
    assets.append({"label": label.strip(), "location": location.strip()})
    entry["assets"] = assets


def add_metric(entry: dict, metric: dict) -> None:
    """Append a {run_url, key, label} wandb metric spec; dedup on (run_url, key)."""
    metrics = [m for m in (entry.get("metrics") or []) if isinstance(m, dict)]
    if not any(m.get("run_url") == metric["run_url"] and m.get("key") == metric["key"] for m in metrics):
        metrics.append(metric)
    entry["metrics"] = metrics


def parse_metric(spec: str) -> dict:
    """'RUN_URL|KEY[|LABEL]' -> {run_url, key, label}; label defaults to the last '/' segment of key."""
    from fetch_metrics import parse_run_url  # deferred: fetch_metrics imports this module

    parts = [part.strip() for part in spec.split("|")]
    assert len(parts) in (2, 3), f"--metric expects 'RUN_URL|KEY[|LABEL]', got: {spec}"
    run_url, key = parts[0], parts[1]
    parse_run_url(run_url)
    assert key, f"--metric needs a non-empty wandb metric key: {spec}"
    label = (parts[2] if len(parts) == 3 else "") or key.rsplit("/", 1)[-1]
    return {"run_url": run_url, "key": key, "label": label}


def parse_asset(spec: str) -> tuple[str, str]:
    """'label=location' -> (label, location); location may itself contain '='."""
    label, sep, location = spec.partition("=")
    assert sep, f"--add-asset expects 'label=location', got: {spec}"
    assert label.strip() and location.strip(), f"--add-asset needs a non-empty label and location: {spec}"
    return label.strip(), location.strip()


def cmd_update(args: argparse.Namespace) -> None:
    root = ledger_root()
    path = entry_path(root, args.id)
    assert path is not None, f"no ledger entry with id {args.id}"
    entry = load_entry(path)
    kind = entry.get("kind", "diff")

    if args.status is not None:
        allowed = statuses_for(kind)
        assert args.status in allowed, f"status for a {kind} must be one of {allowed}"
        entry["status"] = args.status
    if args.title is not None:
        entry["title"] = args.title
    if args.summary is not None:
        entry["summary"] = args.summary
    if args.notes is not None:
        assert kind == "diff", "--notes applies to diff entries"
        entry["notes"] = args.notes
    if args.results is not None:
        assert kind == "experiment", "--results applies to experiment entries"
        entry["results"] = args.results
    if args.conclusion is not None:
        assert kind == "question", "--conclusion applies to question entries"
        entry["conclusion"] = args.conclusion
    if args.question is not None:
        assert kind in ("diff", "experiment"), "--question applies to diff/experiment entries"
        if args.question:
            assert_question_exists(root, args.question)
        entry["question_id"] = args.question or None
    if args.add_diffs:
        assert kind == "experiment", "--add-diffs applies to experiment entries"
        _, diffs_dir, _, _ = subdirs(root)
        new_ids = split_list(args.add_diffs)
        for diff_id in new_ids:
            assert os.path.exists(os.path.join(diffs_dir, f"{diff_id}.json")), f"unknown diff id: {diff_id}"
        append_dedup(entry, "diff_ids", new_ids)
    if args.add_jobs:
        assert kind == "experiment", "--add-jobs applies to experiment entries"
        append_dedup(entry, "job_ids", split_list(args.add_jobs))
    if args.add_wandb:
        assert kind == "experiment", "--add-wandb applies to experiment entries"
        append_dedup(entry, "wandb", split_list(args.add_wandb))
    if args.metric:
        assert kind == "experiment", "--metric applies to experiment entries"
        for spec in args.metric:
            add_metric(entry, parse_metric(spec))
    if args.add_tags:
        append_dedup(entry, "tags", split_list(args.add_tags))
    if args.add_takeaway:
        assert kind == "question", "--add-takeaway applies to question entries"
        takeaways = [t.strip() for t in args.add_takeaway if t.strip()]
        assert takeaways, "--add-takeaway text must not be empty"
        append_dedup(entry, "takeaways", takeaways)
    if args.set_takeaway:
        assert kind == "question", "--set-takeaway applies to question entries"
        assert not args.add_takeaway, "use either --add-takeaway or --set-takeaway, not both"
        entry["takeaways"] = [t.strip() for t in args.set_takeaway if t.strip()]
    if args.add_asset:
        for spec in args.add_asset:
            label, location = parse_asset(spec)
            add_asset(entry, label, location)

    entry["updated"] = now_ts()
    write_entry(path, entry)
    print(entry["id"])


def cmd_set_status(args: argparse.Namespace) -> None:
    root = ledger_root()
    path = entry_path(root, args.id)
    assert path is not None, f"no ledger entry with id {args.id}"
    entry = load_entry(path)
    allowed = statuses_for(entry.get("kind", "diff"))
    assert args.status in allowed, f"status for a {entry.get('kind')} must be one of {allowed}"
    entry["status"] = args.status
    entry["updated"] = now_ts()
    write_entry(path, entry)
    print(entry["id"])


def load_dir_entries(directory: str) -> list[dict]:
    entries = []
    if not os.path.isdir(directory):
        return entries
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        try:
            entries.append(load_entry(os.path.join(directory, name)))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {name}: {exc}", file=sys.stderr)
    return entries


def load_all(root: str) -> list[dict]:
    entries = []
    for d in subdirs(root)[:3]:
        entries += load_dir_entries(d)
    return entries


def where_column(entry: dict) -> str:
    kind = entry.get("kind", "diff")
    if kind == "question":
        return "-"
    where = entry.get("worktree") if kind == "diff" else entry.get("cluster")
    return where or "-"


def cmd_list(args: argparse.Namespace) -> None:
    root = ledger_root()
    entries = load_all(root)
    if args.kind:
        entries = [e for e in entries if e.get("kind") == args.kind]
    if args.status:
        entries = [e for e in entries if e.get("status") == args.status]
    entries.sort(key=lambda e: e.get("updated", ""), reverse=True)
    if not entries:
        print("(no entries)")
        return

    rows = []
    for e in entries:
        where = where_column(e)
        title = e.get("title", "")
        rows.append(
            (
                e.get("id", ""),
                e.get("kind", "")[:4],
                e.get("status", ""),
                where,
                title if len(title) <= 48 else title[:47] + "…",
                e.get("updated", ""),
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(6)]
    for r in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(r)).rstrip())


SEARCH_WEIGHTS = {
    "takeaways": 5,
    "conclusion": 4,
    "title": 3,
    "tags": 3,
    "summary": 2,
    "results": 2,
    "notes": 2,
    "id": 1,
    "cluster": 1,
    "branch": 1,
    "worktree": 1,
    "repo": 1,
    "files": 1,
    "commits": 1,
    "job_ids": 1,
    "wandb": 1,
    "assets": 1,
    "metrics": 1,
}
SNIPPET_FIELDS = ("takeaways", "conclusion", "results", "summary", "notes", "title")
SNIPPET_WIDTH = 140


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def field_text(entry: dict, key: str) -> str:
    """Searchable text of one field; list items (incl. asset dicts) become one line each."""
    value = entry.get(key)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(" ".join(str(v) for v in item.values()))
            else:
                lines.append(str(item))
        return "\n".join(lines)
    return str(value)


def trim_snippet(line: str, width: int = SNIPPET_WIDTH) -> str:
    line = " ".join(line.split())
    return line if len(line) <= width else line[: width - 1] + "…"


def snippets_for(entry: dict, tokens: list[str], limit: int = 3) -> list[str]:
    snippets: list[str] = []
    for key in SNIPPET_FIELDS:
        for line in field_text(entry, key).splitlines():
            if not line.strip():
                continue
            if not any(token in set(tokenize(line)) for token in tokens):
                continue
            snippet = trim_snippet(line)
            if snippet not in snippets:
                snippets.append(snippet)
            if len(snippets) >= limit:
                return snippets
    return snippets


def search_entries(entries: list[dict], query: str, limit: int = 8) -> list[tuple[dict, int, list[str]]]:
    """(entry, score, snippets) for entries matching query, best first. Shared by the CLI and the MCP."""
    tokens = tokenize(query)
    assert tokens, "query must contain at least one alphanumeric token"
    hits = []
    for entry in entries:
        score = 0
        for key, weight in SEARCH_WEIGHTS.items():
            counts = Counter(tokenize(field_text(entry, key)))
            score += weight * sum(counts[token] for token in tokens)
        if score:
            hits.append((entry, score, snippets_for(entry, tokens)))
    hits.sort(key=lambda hit: (hit[1], hit[0].get("updated", "")), reverse=True)
    return hits[:limit]


def cmd_search(args: argparse.Namespace) -> None:
    root = ledger_root()
    entries = load_all(root)
    if args.kind:
        entries = [e for e in entries if e.get("kind") == args.kind]
    hits = search_entries(entries, args.query, args.limit)
    if not hits:
        print("(no matches)")
        return
    for entry, score, snippets in hits:
        head = f"{entry.get('kind', '')}:{entry.get('status', '')}"
        print(f"{score:>4}  {entry.get('id', '')}  {head}  {entry.get('title', '')}")
        for snippet in snippets:
            print(f"        {snippet}")


def cmd_show(args: argparse.Namespace) -> None:
    root = ledger_root()
    path = entry_path(root, args.id)
    assert path is not None, f"no ledger entry with id {args.id}"
    print(json.dumps(load_entry(path), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Central experiment/diff ledger.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add-question", help="record a research question")
    p.add_argument("--title", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--id")
    p.add_argument("--status", default="open")
    p.add_argument("--tags")
    p.set_defaults(func=cmd_add_question)

    p = sub.add_parser("add-diff", help="record a code change")
    p.add_argument("--title", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--id")
    p.add_argument("--status", default="unreviewed")
    p.add_argument("--question", help="research question id this change belongs to")
    p.add_argument("--tags")
    p.add_argument("--commits")
    p.add_argument("--files", help="comma-separated file list, used when --capture is absent")
    p.add_argument("--capture", choices=["working", "branch"])
    p.add_argument("--no-files", action="store_true", help="do not fill files[] from the capture")
    p.set_defaults(func=cmd_add_diff)

    p = sub.add_parser("add-exp", help="record an experiment")
    p.add_argument("--title", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--id")
    p.add_argument("--diffs")
    p.add_argument("--status", default="planned")
    p.add_argument("--question", help="research question id this run belongs to")
    p.add_argument("--cluster")
    p.add_argument("--jobs")
    p.add_argument("--wandb")
    p.add_argument(
        "--metric",
        action="append",
        metavar="RUN_URL|KEY[|LABEL]",
        help="wandb metric to poll and plot; RUN url (.../runs/<id>), repeatable",
    )
    p.add_argument("--tags")
    p.set_defaults(func=cmd_add_exp)

    p = sub.add_parser("update", help="edit an existing entry")
    p.add_argument("id")
    p.add_argument("--status")
    p.add_argument("--title")
    p.add_argument("--summary")
    p.add_argument("--results")
    p.add_argument("--notes")
    p.add_argument("--conclusion", help="questions only: the evolving answer")
    p.add_argument("--question", help="diff/experiment only: question id, empty string clears the link")
    p.add_argument("--add-diffs")
    p.add_argument("--add-jobs")
    p.add_argument("--add-wandb")
    p.add_argument(
        "--metric",
        action="append",
        metavar="RUN_URL|KEY[|LABEL]",
        help="experiments only: attach a wandb metric to poll and plot; repeatable, deduped",
    )
    p.add_argument("--add-tags")
    p.add_argument(
        "--add-takeaway",
        action="append",
        help="questions only: a short self-contained finding; repeatable",
    )
    p.add_argument(
        "--set-takeaway",
        action="append",
        help="questions only: REPLACE the takeaways list with these; repeatable (for slimming rewrites)",
    )
    p.add_argument(
        "--add-asset",
        action="append",
        metavar="LABEL=LOCATION",
        help="attach an artifact (checkpoint, video, plot, wandb url); repeatable",
    )
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("set-status", help="shorthand for update --status")
    p.add_argument("id")
    p.add_argument("status")
    p.set_defaults(func=cmd_set_status)

    p = sub.add_parser("list", help="compact table of entries")
    p.add_argument("--kind", choices=["question", "diff", "experiment"])
    p.add_argument("--status")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("search", help="rank entries against a query (takeaways weighted highest)")
    p.add_argument("query")
    p.add_argument("--kind", choices=["question", "diff", "experiment"])
    p.add_argument("--limit", type=int, default=8)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("show", help="pretty-print one entry")
    p.add_argument("id")
    p.set_defaults(func=cmd_show)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
