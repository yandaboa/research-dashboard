"""Stdio MCP server exposing the experiment/diff ledger to agents in any worktree."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ledger import (  # noqa: E402
    EXP_STATUSES,
    QUESTION_STATUSES,
    append_dedup,
    assert_question_exists,
    atomic_write,
    capture_diff,
    detect_worktree,
    ensure_root,
    entry_path,
    ledger_root,
    load_all,
    load_entry,
    now_ts,
    slugify,
    subdirs,
    unique_id,
    where_column,
    write_entry,
)

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "uwlab-ledger", "version": "1.0.0"}

GUIDE = """UWLab ledger — what to write and how.

The ledger is a shared record of research questions and the work done under them. It lives outside
any worktree, so entries written from one worktree are visible everywhere.

HIERARCHY
  question -> diffs + experiments. A question is the research line; every diff and experiment
  should carry the question_id of the line it belongs to.

ENTRY KINDS
  question    one research line, long-lived. Fields: id, title, summary, status, conclusion,
              tags[]. status: open | answered | parked -- agents keep this current.
              conclusion: the evolving answer, updated as evidence comes in.
  diff        one coherent code change. Fields: id, title, summary, question_id, repo, worktree,
              branch, commits[], files[], patch (stored under patches/), status, notes, tags[].
              status: unreviewed | questionable | known_good  -- SET BY THE USER, not by agents.
              notes: also the user's. Agents never write status or notes on a diff.
  experiment  one run (or one tightly-coupled set of runs). Fields: id, title, summary,
              question_id, diff_ids[], status, cluster, job_ids[], wandb[], results, tags[].
              status: planned | running | done | killed | failed  -- agents keep this current.

WORKFLOW
  1. Find or create the question: ledger_list(kind="question") first; reuse an existing one if the
     work belongs to that line. Only call ledger_add_question for a genuinely new line.
  2. ledger_add_diff(..., question_id=<q-id>) when a coherent change is complete.
  3. ledger_add_experiment(..., question_id=<q-id>, diff_ids=[...]) when the run is launched.
  4. ledger_update the experiment with status + results as jobs finish or die.
  5. When the evidence is in, ledger_update the question's conclusion (and status="answered" if it
     is settled, "parked" if the line is dropped).

CONVENTIONS
  * One question per research line, not per run. Near-duplicate questions make the dashboard
    useless -- search before creating.
  * One ledger_add_diff per coherent change, written when the change is complete, not per file.
    summary is required: say what changed and why, in a couple of sentences an outside reader can
    follow. Do not paste the diff -- the patch is captured for you.
  * repo_dir must be the absolute path of the worktree the change lives in. Git repo, branch and
    the patch are captured from there.
  * Always link an experiment to the diff(s) it exercises via diff_ids. An experiment with no diff
    is only correct when nothing in the tree changed.
  * Keep experiment status current. A stale "running" for a job that died days ago is worse than
    no entry at all -- update to done/killed/failed as soon as you know, and put the numbers in
    results (success rate, iterations, what it showed).
  * Job ids, wandb run urls/ids and cluster go on the experiment, not in prose.
  * Never set status or notes on a diff entry: review is the user's job, in the GUI.

EXAMPLE SEQUENCE
  1. ledger_add_question(title="Does a shared-trunk critic fix value collapse on PC obs?",
                         summary="Privileged critics collapse on point-cloud observations in "
                                 "in-context PPO. Does sharing the actor trunk fix it?",
                         tags=["rl","in-context"])
     -> q-does-a-shared-trunk-critic-fix-value-collapse-on-pc-obs
  2. ledger_add_diff(title="Shared-trunk critic for in-context PPO",
                     summary="Critic now shares the actor trunk instead of taking privileged obs; "
                             "privileged critics collapsed on PC obs. Adds critic_design flag.",
                     repo_dir="/home/yandabao/UWLab-patrick-private/.claude/worktrees/incontext",
                     question_id="q-does-a-shared-trunk-critic-fix-value-collapse-on-pc-obs",
                     capture="working", tags=["rl","in-context"])
     -> 2026-08-31-shared-trunk-critic-for-in-context-ppo
  3. ledger_add_experiment(title="ctx16 PC bias, shared-trunk critic",
                           summary="Two seeds on Tillicum, 16-step context, obs-bias POMDP; tests "
                                   "whether the shared trunk fixes the value collapse.",
                           question_id="q-does-a-shared-trunk-critic-fix-value-collapse-on-pc-obs",
                           diff_ids=["2026-08-31-shared-trunk-critic-for-in-context-ppo"],
                           status="running", cluster="tillicum", job_ids=["265935","265937"],
                           wandb=["yandabaocs-university-of-washington/.../265935"])
     -> exp-2026-08-31-ctx16-pc-bias-shared-trunk-critic
  4. when the runs finish:
     ledger_update(id="exp-2026-08-31-ctx16-pc-bias-shared-trunk-critic", status="done",
                   results="0.95 / 0.97 success at iter ~4k; no value collapse. Killed at 6k.")
  5. when the question is settled:
     ledger_update(id="q-does-a-shared-trunk-critic-fix-value-collapse-on-pc-obs",
                   status="answered",
                   conclusion="Yes. Shared trunk + LR warmup holds 0.95-0.97 where the privileged "
                              "critic collapsed within 500 iters.")

TOOLS
  ledger_guide           this text
  ledger_add_question    open a research line (find-or-create; check ledger_list first)
  ledger_add_diff        record a code change (captures the patch), linked to a question
  ledger_add_experiment  record a run, linked to a question and diff ids
  ledger_update          edit an entry / set conclusion / append job ids, wandb, tags, diff ids
  ledger_list            browse entries (filter by kind/status/worktree/query)
  ledger_show            full JSON for one entry (+ patch path for diffs)
"""

TOOLS = [
    {
        "name": "ledger_guide",
        "description": (
            "Read this FIRST before writing to the experiment/diff ledger: entry schemas, status enums, "
            "conventions, and a worked example of what a good entry looks like. No arguments."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "ledger_add_question",
        "description": (
            "Open a research question — the TOP-LEVEL entity that diffs and experiments hang off. Create ONE "
            "question per research line and reuse it for everything in that line: call ledger_list(kind="
            "'question') FIRST and pass the existing id instead of creating a near-duplicate. A question is "
            "long-lived (weeks), not per-run. Returns the new question id."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "the question itself, phrased as a question"},
                "summary": {"type": "string", "description": "motivation and context, a few sentences"},
                "status": {"type": "string", "enum": list(QUESTION_STATUSES)},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "summary"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ledger_add_diff",
        "description": (
            "Record a completed code change in the ledger and capture its patch. Call once per coherent "
            "change (not per file), after the edits are done, and before logging any experiment that "
            "depends on it. Pass the question_id of the research line it belongs to — real research work is "
            "expected to be linked to a question. Returns the new diff id to pass to ledger_add_experiment. "
            "Review status and notes are the user's and cannot be set here."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "short imperative title of the change"},
                "summary": {
                    "type": "string",
                    "description": "what changed and why, a couple of sentences; do not paste the diff",
                },
                "repo_dir": {
                    "type": "string",
                    "description": "absolute path of the worktree/checkout the change lives in",
                },
                "question_id": {
                    "type": "string",
                    "description": "id from ledger_add_question; validated to exist",
                },
                "capture": {
                    "type": "string",
                    "enum": ["working", "branch"],
                    "description": "working = uncommitted diff vs HEAD (default); branch = merge-base...HEAD",
                },
                "tags": {"type": "array", "items": {"type": "string"}},
                "commits": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "summary", "repo_dir"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ledger_add_experiment",
        "description": (
            "Record an experiment (a run or a tightly-coupled set of runs) and link it to the research "
            "question it answers and the diff ids it exercises. Call when a run is launched, with "
            "status='running'; then keep it current with ledger_update as jobs finish or die. A question_id "
            "is expected for real research work. Returns the new experiment id."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string", "description": "what this run tests and how it is set up"},
                "question_id": {
                    "type": "string",
                    "description": "id from ledger_add_question; validated to exist",
                },
                "diff_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "ids from ledger_add_diff; validated to exist",
                },
                "status": {"type": "string", "enum": list(EXP_STATUSES)},
                "cluster": {"type": "string", "description": "e.g. local, hyak, tillicum"},
                "job_ids": {"type": "array", "items": {"type": "string"}},
                "wandb": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "summary"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ledger_update",
        "description": (
            "Update an existing ledger entry. Main uses: move an experiment to done/killed/failed and fill "
            "in results when runs finish -- a stale 'running' is worse than no entry -- and write the "
            "conclusion (and status) of a question once the evidence is in. Also relinks a diff/experiment "
            "to a question and appends job ids, wandb runs, tags and diff links. Diff review status and "
            "notes are set by the user in the GUI and are rejected here."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": sorted(set(EXP_STATUSES) | set(QUESTION_STATUSES)),
                    "description": "experiments and questions only; diff review status is the user's",
                },
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "results": {"type": "string", "description": "experiments only: numbers and what they showed"},
                "conclusion": {"type": "string", "description": "questions only: the answer so far, and why"},
                "question_id": {
                    "type": "string",
                    "description": "diff/experiment only: question to link to; empty string clears the link",
                },
                "add_diff_ids": {"type": "array", "items": {"type": "string"}},
                "add_job_ids": {"type": "array", "items": {"type": "string"}},
                "add_wandb": {"type": "array", "items": {"type": "string"}},
                "add_tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ledger_list",
        "description": (
            "Browse ledger entries as a compact table (most recently updated first). Use it to find the "
            "existing question for a research line before creating a new one, to find the diff id for a "
            "change you just made, to see which experiments are still marked running, or to check whether "
            "something has already been recorded."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["question", "diff", "experiment"]},
                "status": {"type": "string"},
                "worktree": {"type": "string", "description": "diff entries only: worktree name"},
                "query": {"type": "string", "description": "case-insensitive substring over id/title/summary/tags"},
                "limit": {"type": "integer", "minimum": 1, "description": "default 20"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "ledger_show",
        "description": (
            "Show one ledger entry as full JSON. For diffs it also prints the absolute path of the stored "
            "patch so you can Read it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    },
]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def as_str(args: dict, key: str, required: bool = False, default: str | None = None) -> str | None:
    value = args.get(key, None)
    if value is None:
        assert not required, f"{key} is required"
        return default
    assert isinstance(value, str), f"{key} must be a string"
    return value


def as_list(args: dict, key: str) -> list[str]:
    value = args.get(key, None)
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    assert isinstance(value, list), f"{key} must be an array of strings"
    for item in value:
        assert isinstance(item, str), f"{key} must be an array of strings"
    return [item.strip() for item in value if item.strip()]


def assert_diffs_exist(root: str, diff_ids: list[str]) -> None:
    _, diffs_dir, _, _ = subdirs(root)
    for diff_id in diff_ids:
        assert os.path.exists(os.path.join(diffs_dir, f"{diff_id}.json")), f"unknown diff id: {diff_id}"


def tool_guide(args: dict) -> str:
    return GUIDE


def tool_add_question(args: dict) -> str:
    root = ledger_root()
    ensure_root(root)
    title = as_str(args, "title", required=True)
    summary = as_str(args, "summary", required=True)
    assert summary.strip(), "summary must not be empty: say why this question matters"
    status = as_str(args, "status", default="open")
    assert status in QUESTION_STATUSES, f"status must be one of {QUESTION_STATUSES}"

    questions_dir, _, _, _ = subdirs(root)
    entry_id = unique_id(root, f"q-{slugify(title)}")

    ts = now_ts()
    entry = {
        "kind": "question",
        "id": entry_id,
        "title": title,
        "summary": summary,
        "status": status,
        "conclusion": "",
        "tags": as_list(args, "tags"),
        "created": ts,
        "updated": ts,
    }
    write_entry(os.path.join(questions_dir, f"{entry_id}.json"), entry)
    return entry_id


def tool_add_diff(args: dict) -> str:
    root = ledger_root()
    ensure_root(root)
    title = as_str(args, "title", required=True)
    summary = as_str(args, "summary", required=True)
    repo_dir = as_str(args, "repo_dir", required=True)
    capture = as_str(args, "capture", default="working")
    assert capture in ("working", "branch"), "capture must be 'working' or 'branch'"
    assert os.path.isabs(repo_dir), "repo_dir must be an absolute path"
    assert os.path.isdir(repo_dir), f"repo_dir does not exist: {repo_dir}"
    assert summary.strip(), "summary must not be empty: say what changed and why"
    question_id = as_str(args, "question_id", default="") or ""
    if question_id:
        assert_question_exists(root, question_id)

    _, diffs_dir, _, _ = subdirs(root)
    entry_id = unique_id(root, f"{datetime.now().strftime('%Y-%m-%d')}-{slugify(title)}")

    repo, worktree, branch = detect_worktree(repo_dir)
    patch_text, changed = capture_diff(capture, repo_dir)
    patch_rel = None
    if patch_text.strip():
        patch_rel = os.path.join("patches", f"{entry_id}.patch")
        atomic_write(os.path.join(root, patch_rel), patch_text + "\n")

    ts = now_ts()
    entry = {
        "kind": "diff",
        "id": entry_id,
        "title": title,
        "summary": summary,
        "question_id": question_id or None,
        "repo": repo,
        "worktree": worktree,
        "branch": branch,
        "commits": as_list(args, "commits"),
        "files": changed,
        "patch": patch_rel,
        "status": "unreviewed",
        "notes": "",
        "tags": as_list(args, "tags"),
        "created": ts,
        "updated": ts,
    }
    write_entry(os.path.join(diffs_dir, f"{entry_id}.json"), entry)
    stored = "yes" if patch_rel else "no (empty diff)"
    return f"{entry_id}\nrecorded in {repo}/{worktree} ({branch}); patch stored: {stored}; {len(changed)} files"


def tool_add_experiment(args: dict) -> str:
    root = ledger_root()
    ensure_root(root)
    title = as_str(args, "title", required=True)
    summary = as_str(args, "summary", required=True)
    assert summary.strip(), "summary must not be empty: say what this run tests"
    status = as_str(args, "status", default="planned")
    assert status in EXP_STATUSES, f"status must be one of {EXP_STATUSES}"

    question_id = as_str(args, "question_id", default="") or ""
    if question_id:
        assert_question_exists(root, question_id)
    diff_ids = as_list(args, "diff_ids")
    assert_diffs_exist(root, diff_ids)

    _, _, exp_dir, _ = subdirs(root)
    entry_id = unique_id(root, f"exp-{datetime.now().strftime('%Y-%m-%d')}-{slugify(title)}")

    ts = now_ts()
    entry = {
        "kind": "experiment",
        "id": entry_id,
        "title": title,
        "summary": summary,
        "question_id": question_id or None,
        "diff_ids": diff_ids,
        "status": status,
        "cluster": as_str(args, "cluster", default="") or "",
        "job_ids": as_list(args, "job_ids"),
        "wandb": as_list(args, "wandb"),
        "results": "",
        "tags": as_list(args, "tags"),
        "created": ts,
        "updated": ts,
    }
    write_entry(os.path.join(exp_dir, f"{entry_id}.json"), entry)
    return entry_id


def tool_update(args: dict) -> str:
    root = ledger_root()
    entry_id = as_str(args, "id", required=True)
    path = entry_path(root, entry_id)
    assert path is not None, f"no ledger entry with id {entry_id}"
    entry = load_entry(path)
    kind = entry.get("kind", "diff")

    status = as_str(args, "status")
    if status is not None:
        assert kind != "diff", (
            "review status and notes on a diff are set by the user in the ledger GUI; agents cannot change them"
        )
        allowed = QUESTION_STATUSES if kind == "question" else EXP_STATUSES
        assert status in allowed, f"status for a {kind} must be one of {allowed}"
        entry["status"] = status
    title = as_str(args, "title")
    if title is not None:
        entry["title"] = title
    summary = as_str(args, "summary")
    if summary is not None:
        entry["summary"] = summary
    results = as_str(args, "results")
    if results is not None:
        assert kind == "experiment", "results applies to experiment entries"
        entry["results"] = results
    conclusion = as_str(args, "conclusion")
    if conclusion is not None:
        assert kind == "question", "conclusion applies to question entries"
        entry["conclusion"] = conclusion
    question_id = as_str(args, "question_id")
    if question_id is not None:
        assert kind in ("diff", "experiment"), "question_id applies to diff/experiment entries"
        if question_id:
            assert_question_exists(root, question_id)
        entry["question_id"] = question_id or None

    add_diff_ids = as_list(args, "add_diff_ids")
    if add_diff_ids:
        assert kind == "experiment", "add_diff_ids applies to experiment entries"
        assert_diffs_exist(root, add_diff_ids)
        append_dedup(entry, "diff_ids", add_diff_ids)
    add_job_ids = as_list(args, "add_job_ids")
    if add_job_ids:
        assert kind == "experiment", "add_job_ids applies to experiment entries"
        append_dedup(entry, "job_ids", add_job_ids)
    add_wandb = as_list(args, "add_wandb")
    if add_wandb:
        assert kind == "experiment", "add_wandb applies to experiment entries"
        append_dedup(entry, "wandb", add_wandb)
    add_tags = as_list(args, "add_tags")
    if add_tags:
        append_dedup(entry, "tags", add_tags)

    entry["updated"] = now_ts()
    write_entry(path, entry)
    return json.dumps(entry, indent=2)


def matches_query(entry: dict, query: str) -> bool:
    haystack = " ".join(
        [
            str(entry.get("id", "")),
            str(entry.get("title", "")),
            str(entry.get("summary", "")),
            " ".join(entry.get("tags") or []),
        ]
    ).lower()
    return query.lower() in haystack


def tool_list(args: dict) -> str:
    root = ledger_root()
    kind = as_str(args, "kind")
    assert kind in (None, "question", "diff", "experiment"), "kind must be 'question', 'diff' or 'experiment'"
    status = as_str(args, "status")
    worktree = as_str(args, "worktree")
    query = as_str(args, "query")
    limit = args.get("limit", 20)
    assert isinstance(limit, int) and not isinstance(limit, bool), "limit must be an integer"
    assert limit > 0, "limit must be positive"

    entries = load_all(root)
    if kind:
        entries = [e for e in entries if e.get("kind") == kind]
    if status:
        entries = [e for e in entries if e.get("status") == status]
    if worktree:
        entries = [e for e in entries if e.get("worktree") == worktree]
    if query:
        entries = [e for e in entries if matches_query(e, query)]
    entries.sort(key=lambda e: e.get("updated", ""), reverse=True)
    if not entries:
        return "(no entries)"

    total = len(entries)
    shown = entries[:limit]
    rows = []
    for e in shown:
        title = e.get("title", "")
        rows.append(
            (
                e.get("id", ""),
                e.get("kind", "")[:4],
                e.get("status", ""),
                where_column(e),
                title if len(title) <= 48 else title[:47] + "…",
                e.get("updated", ""),
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(6)]
    lines = ["  ".join(value.ljust(widths[i]) for i, value in enumerate(r)).rstrip() for r in rows]
    if total > len(shown):
        lines.append(f"{len(shown)} of {total} shown")
    return "\n".join(lines)


def tool_show(args: dict) -> str:
    root = ledger_root()
    entry_id = as_str(args, "id", required=True)
    path = entry_path(root, entry_id)
    assert path is not None, f"no ledger entry with id {entry_id}"
    entry = load_entry(path)
    text = json.dumps(entry, indent=2)
    rel = entry.get("patch")
    if rel:
        patch_path = os.path.join(root, rel)
        if os.path.exists(patch_path):
            text += f"\npatch: {patch_path}"
    return text


HANDLERS = {
    "ledger_guide": tool_guide,
    "ledger_add_question": tool_add_question,
    "ledger_add_diff": tool_add_diff,
    "ledger_add_experiment": tool_add_experiment,
    "ledger_update": tool_update,
    "ledger_list": tool_list,
    "ledger_show": tool_show,
}


def call_tool(name: str, args: dict) -> dict:
    handler = HANDLERS.get(name)
    if handler is None:
        return {"content": [{"type": "text", "text": f"error: unknown tool: {name}"}], "isError": True}
    try:
        text = handler(args if isinstance(args, dict) else {})
    except Exception as exc:  # never let a bad call kill the loop
        message = str(exc) or exc.__class__.__name__
        log(f"tool {name} failed: {exc.__class__.__name__}: {message}")
        return {"content": [{"type": "text", "text": f"error: {message}"}], "isError": True}
    return {"content": [{"type": "text", "text": text}], "isError": False}


def handle_request(msg: dict) -> dict | None:
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if isinstance(method, str) and method.startswith("notifications/"):
        return None

    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if isinstance(requested, str) else PROTOCOL_VERSION
        result = {"protocolVersion": version, "capabilities": {"tools": {}}, "serverInfo": SERVER_INFO}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        result = call_tool(params.get("name"), params.get("arguments") or {})
    else:
        if msg_id is None:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"unknown method: {method}"}}

    if msg_id is None:
        return None
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def serve(stdin, stdout) -> None:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            log(f"malformed json line: {exc}")
            continue
        if not isinstance(msg, dict):
            log("ignoring non-object json message")
            continue
        try:
            response = handle_request(msg)
        except Exception as exc:  # a broken request must not end the session
            log(f"request handling failed: {exc.__class__.__name__}: {exc}")
            msg_id = msg.get("id")
            if msg_id is None:
                continue
            response = {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32603, "message": str(exc)}}
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


def main() -> None:
    log(f"uwlab-ledger mcp server: root={ledger_root()}")
    serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
