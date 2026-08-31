"""Append/update entries in the central experiment + diff ledger shared by all worktrees."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DIFF_STATUSES = ("unreviewed", "questionable", "known_good")
EXP_STATUSES = ("planned", "running", "done", "killed", "failed")


def ledger_root() -> str:
    return os.environ.get("UWLAB_LEDGER_ROOT", DEFAULT_ROOT)


def subdirs(root: str) -> tuple[str, str, str]:
    return (
        os.path.join(root, "diffs"),
        os.path.join(root, "experiments"),
        os.path.join(root, "patches"),
    )


def ensure_root(root: str) -> None:
    for d in subdirs(root):
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
    diffs_dir, exp_dir, _ = subdirs(root)
    for d in (diffs_dir, exp_dir):
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
    return DIFF_STATUSES if kind == "diff" else EXP_STATUSES


def cmd_add_diff(args: argparse.Namespace) -> None:
    root = ledger_root()
    ensure_root(root)
    assert args.status in DIFF_STATUSES, f"status must be one of {DIFF_STATUSES}"
    diffs_dir, _, patches_dir = subdirs(root)

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
        "repo": repo,
        "worktree": worktree,
        "branch": branch,
        "commits": split_list(args.commits),
        "files": files,
        "patch": patch_rel,
        "status": args.status,
        "notes": "",
        "tags": split_list(args.tags),
        "created": ts,
        "updated": ts,
    }
    write_entry(os.path.join(diffs_dir, f"{entry_id}.json"), entry)
    print(entry_id)


def cmd_add_exp(args: argparse.Namespace) -> None:
    root = ledger_root()
    ensure_root(root)
    assert args.status in EXP_STATUSES, f"status must be one of {EXP_STATUSES}"
    diffs_dir, exp_dir, _ = subdirs(root)

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
        "diff_ids": diff_ids,
        "status": args.status,
        "cluster": args.cluster or "",
        "job_ids": split_list(args.jobs),
        "wandb": split_list(args.wandb),
        "results": "",
        "tags": split_list(args.tags),
        "created": ts,
        "updated": ts,
    }
    write_entry(os.path.join(exp_dir, f"{entry_id}.json"), entry)
    print(entry_id)


def append_dedup(entry: dict, key: str, values: list[str]) -> None:
    existing = list(entry.get(key) or [])
    for value in values:
        if value not in existing:
            existing.append(value)
    entry[key] = existing


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
    if args.add_diffs:
        assert kind == "experiment", "--add-diffs applies to experiment entries"
        diffs_dir, _, _ = subdirs(root)
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
    if args.add_tags:
        append_dedup(entry, "tags", split_list(args.add_tags))

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


def load_all(root: str) -> list[dict]:
    entries = []
    for d in subdirs(root)[:2]:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            try:
                entries.append(load_entry(os.path.join(d, name)))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"warning: skipping {name}: {exc}", file=sys.stderr)
    return entries


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
        where = e.get("worktree") if e.get("kind") == "diff" else e.get("cluster")
        title = e.get("title", "")
        rows.append(
            (
                e.get("id", ""),
                e.get("kind", "")[:4],
                e.get("status", ""),
                where or "-",
                title if len(title) <= 48 else title[:47] + "…",
                e.get("updated", ""),
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(6)]
    for r in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(r)).rstrip())


def cmd_show(args: argparse.Namespace) -> None:
    root = ledger_root()
    path = entry_path(root, args.id)
    assert path is not None, f"no ledger entry with id {args.id}"
    print(json.dumps(load_entry(path), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Central experiment/diff ledger.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add-diff", help="record a code change")
    p.add_argument("--title", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--id")
    p.add_argument("--status", default="unreviewed")
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
    p.add_argument("--cluster")
    p.add_argument("--jobs")
    p.add_argument("--wandb")
    p.add_argument("--tags")
    p.set_defaults(func=cmd_add_exp)

    p = sub.add_parser("update", help="edit an existing entry")
    p.add_argument("id")
    p.add_argument("--status")
    p.add_argument("--title")
    p.add_argument("--summary")
    p.add_argument("--results")
    p.add_argument("--notes")
    p.add_argument("--add-diffs")
    p.add_argument("--add-jobs")
    p.add_argument("--add-wandb")
    p.add_argument("--add-tags")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("set-status", help="shorthand for update --status")
    p.add_argument("id")
    p.add_argument("status")
    p.set_defaults(func=cmd_set_status)

    p = sub.add_parser("list", help="compact table of entries")
    p.add_argument("--kind", choices=["diff", "experiment"])
    p.add_argument("--status")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="pretty-print one entry")
    p.add_argument("id")
    p.set_defaults(func=cmd_show)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
