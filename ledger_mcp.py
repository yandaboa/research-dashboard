"""Stdio MCP server exposing the experiment/diff ledger to agents in any worktree."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_metrics import parse_run_url  # noqa: E402
from ledger import (  # noqa: E402
    EXP_STATUSES,
    QUESTION_STATUSES,
    add_asset,
    add_metric,
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
    search_entries,
    slugify,
    subdirs,
    unique_id,
    where_column,
    write_entry,
)

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "uwlab-ledger", "version": "1.0.0"}

SEARCH_DIRECTIVE = (
    "NEXT STEP (required): spawn a background subagent (e.g. the Explore agent) to search this ledger for "
    "prior knowledge relevant to this work — have it call the `ledger_search` MCP tool (or run "
    "`python3 /home/yandabao/ledger/ledger.py search '<terms>'`) with 2-3 phrasings of the topic, read the "
    "takeaways of any matching questions, and report back anything that should change your plan. Incorporate "
    "relevant takeaways before proceeding."
)

WANDB_DIRECTIVE = (
    "wandb: real experiments must ALWAYS log to wandb (--logger wandb). Smoke tests and quick probes must NEVER "
    "be logged to wandb. Once the run is up, attach its RUN url (…/runs/<id>) and 1-3 key metric names via "
    "ledger_update add_metrics — the dashboard polls them every 5 minutes and plots them."
)

METRICS_MISSING = (
    "metrics missing: this run is marked running but has no linked wandb metrics — attach them with "
    "ledger_update(id=..., add_metrics=[{run_url, key, label}]) as soon as the run reports."
)

BREVITY_LIMITS = {"title": 70, "summary": 240, "results": 240, "conclusion": 400, "takeaway": 160, "notes": 240}


def brevity_warnings(args: dict) -> list[str]:
    """Length warnings for prose fields — the user reads dozens of entries; verbosity is a defect."""
    warnings = []
    for field in ("title", "summary", "results", "conclusion"):
        value = args.get(field)
        if isinstance(value, str) and len(value) > BREVITY_LIMITS[field]:
            warnings.append(
                f"TOO VERBOSE: {field} is {len(value)} chars (limit ~{BREVITY_LIMITS[field]}). "
                f"Rewrite it shorter with ledger_update — the user scans dozens of entries."
            )
    takeaways = args.get("add_takeaways")
    if isinstance(takeaways, list):
        long = [t for t in takeaways if isinstance(t, str) and len(t) > BREVITY_LIMITS["takeaway"]]
        if long:
            warnings.append(
                f"TOO VERBOSE: {len(long)} takeaway(s) exceed ~{BREVITY_LIMITS['takeaway']} chars. "
                "A takeaway is ONE short declarative fact."
            )
    return warnings

GUIDE = """UWLab ledger — what to write and how.

The ledger is a shared record of research questions and the work done under them. It lives outside
any worktree, so entries written from one worktree are visible everywhere. It is agent-driven: the
user hands a session a research question or an experiment, and the session finds-or-creates the
question, attaches the work to it, and keeps it current.

BREVITY (hard rule — the user scans dozens of entries every morning)
  Every prose field is telegraphic. Cut articles, hedges, restated context, and anything the
  structured fields already say (ids, jobs, clusters, urls go in their fields, never in prose).
  Limits (tools warn past them):
    title       <= ~8 words, no trailing clause
    summary     1-2 short sentences (<= ~240 chars): what + why, nothing else
    results     numbers first, one clause of interpretation (<= ~240 chars)
    conclusion  <= 3 short sentences
    takeaway    ONE declarative fact, <= ~160 chars
  Write like a lab notebook margin, not a report.

HIERARCHY
  question -> diffs + experiments + assets. A question is the research line; every diff and
  experiment should carry the question_id of the line it belongs to.

ENTRY KINDS
  question    one research line, long-lived. Fields: id, title, summary, status, conclusion,
              takeaways[], tags[], assets[]. status: open | answered | parked -- agents keep this
              current. conclusion: the evolving answer narrative, updated as evidence comes in.
              takeaways: the stable distilled findings (see TAKEAWAYS below).
  diff        one coherent code change. Fields: id, title, summary, question_id, repo, worktree,
              branch, commits[], files[], patch (stored under patches/), status, notes, tags[],
              assets[]. status: unreviewed | questionable | known_good -- SET BY THE USER, not by
              agents. notes: also the user's. Agents never write status or notes on a diff.
  experiment  one run (or one tightly-coupled set of runs). Fields: id, title, summary,
              question_id, diff_ids[], status, cluster, job_ids[], wandb[], results, tags[],
              assets[]. status: planned | running | done | killed | failed -- agents keep current.

LIFECYCLE
  1. The user hands the session a research question or an experiment to run.
  2. SEARCH FIRST: ledger_search with 2-3 phrasings of the topic before doing any work. Read the
     takeaways of the matching questions -- they are prior findings that should change your plan.
     Best practice: spawn a background subagent (the Explore agent) to run the searches and report
     back, so the searching does not eat the main thread's context.
  3. Find or create the question. Reuse the existing question if the work belongs to that line;
     ledger_add_question only for a genuinely new line.
  4. Attach the work as it is produced: ledger_add_diff(question_id=...) per coherent change,
     ledger_add_experiment(question_id=..., diff_ids=[...]) when a run is launched, and
     ledger_add_asset for every artifact worth finding again (checkpoint, video, plot, dataset,
     wandb run, log dir).
  5. Keep it current: ledger_update the experiment's status + results as jobs finish or die, and
     the question's conclusion as evidence comes in (status="answered" when settled, "parked" if
     the line is dropped).
  6. When a finding stabilizes, distill it into takeaways with
     ledger_update(add_takeaways=[...]).
  7. Review status and notes on diffs are the user's, set in the GUI. Never write them.

TAKEAWAYS
  Takeaways are the ledger's knowledge base and what future agents search against. Each one is a
  single, short, declarative fact, self-contained enough that an agent with zero context can act on
  it: name the mechanism/setting and the condition it holds under.
    good: "Privileged critic collapses on PC obs in in-context PPO; shared-trunk + LR warmup holds 0.95+."
    good: "Warm-cache camera envs need HF_HUB_OFFLINE=1 or first reset stalls ~15 min on texture HEADs."
    bad:  "It worked." / "The fix helped." / "See exp-2026-08-31-..." (no context, not searchable)
    bad:  anything over ~160 chars — split it or cut it
  Grow the list with add_takeaways; rewrite the whole list (e.g. to slim it) with takeaways=[...].
  conclusion is the narrative; a takeaway is the fact extracted from it.

ASSETS
  {label, location} on any entry. location is a path, URL, wandb link, checkpoint, zarr, video.
  Attach checkpoints and plots to the experiment that produced them, dataset/paper links to the
  question. Dedup is on location.

METRICS
  Real experiments must ALWAYS log to wandb (--logger wandb); smoke tests and quick probes must
  NEVER be logged to wandb. Once a run is up, attach its RUN url (.../runs/<id>, not a project url)
  and 1-3 metric keys to the experiment -- metrics=[{run_url, key, label}] on
  ledger_add_experiment, or add_metrics=[...] on ledger_update. A poller refreshes them every 5
  minutes and the dashboard plots each metric under the experiment. Pick the 1-3 metrics that
  answer the research question (e.g. the success rate the run is meant to move), not every logged
  scalar; label is a short human name, defaulting to the last '/' segment of the key.

CONVENTIONS
  * One question per research line, not per run. Near-duplicate questions make the dashboard
    useless -- ledger_search before creating.
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

EXAMPLE SEQUENCE
  1. ledger_search(query="privileged critic value collapse point cloud PPO")
     -> read the takeaways of any hits before planning
  2. ledger_add_question(title="Does a shared-trunk critic fix value collapse on PC obs?",
                         summary="Privileged critics collapse on point-cloud observations in "
                                 "in-context PPO. Does sharing the actor trunk fix it?",
                         tags=["rl","in-context"])
     -> q-does-a-shared-trunk-critic-fix-value-collapse-on-pc-obs
  3. ledger_add_diff(title="Shared-trunk critic for in-context PPO",
                     summary="Critic now shares the actor trunk instead of taking privileged obs; "
                             "privileged critics collapsed on PC obs. Adds critic_design flag.",
                     repo_dir="/home/yandabao/UWLab-patrick-private/.claude/worktrees/incontext",
                     question_id="q-does-a-shared-trunk-critic-fix-value-collapse-on-pc-obs",
                     capture="working", tags=["rl","in-context"])
     -> 2026-08-31-shared-trunk-critic-for-in-context-ppo
  4. ledger_add_experiment(title="ctx16 PC bias, shared-trunk critic",
                           summary="Two seeds on Tillicum, 16-step context, obs-bias POMDP; tests "
                                   "whether the shared trunk fixes the value collapse.",
                           question_id="q-does-a-shared-trunk-critic-fix-value-collapse-on-pc-obs",
                           diff_ids=["2026-08-31-shared-trunk-critic-for-in-context-ppo"],
                           status="running", cluster="tillicum", job_ids=["265935","265937"],
                           wandb=["yandabaocs-university-of-washington/.../265935"],
                           metrics=[{"run_url": "https://wandb.ai/<entity>/<project>/runs/<run_id>",
                                     "key": "Curriculum/pomdps/mean_success_rate",
                                     "label": "success"}])
     -> exp-2026-08-31-ctx16-pc-bias-shared-trunk-critic
  5. as artifacts appear:
     ledger_add_asset(id="exp-2026-08-31-ctx16-pc-bias-shared-trunk-critic",
                      label="ckpt iter 6000", location="/home/yandabao/pulled_ckpts/265935_6000.pt")
  6. when the runs finish:
     ledger_update(id="exp-2026-08-31-ctx16-pc-bias-shared-trunk-critic", status="done",
                   results="0.95 / 0.97 success at iter ~4k; no value collapse. Killed at 6k.")
  7. when the question is settled:
     ledger_update(id="q-does-a-shared-trunk-critic-fix-value-collapse-on-pc-obs",
                   status="answered",
                   conclusion="Yes. Shared trunk + LR warmup holds 0.95-0.97 where the privileged "
                              "critic collapsed within 500 iters.",
                   add_takeaways=["In-context PPO on point-cloud obs needs a shared-trunk critic: "
                                  "a privileged critic collapses within ~500 iters, shared trunk "
                                  "plus LR warmup holds 0.95-0.97 success."])

TOOLS
  ledger_guide           this text
  ledger_search          full-text search over questions (takeaways weighted highest), diffs, exps
  ledger_add_question    open a research line (search first; find-or-create)
  ledger_add_diff        record a code change (captures the patch), linked to a question
  ledger_add_experiment  record a run, linked to a question and diff ids
  ledger_add_asset       attach an artifact (checkpoint, video, plot, wandb) to any entry
  ledger_update          status/results/conclusion, add_takeaways, add_metrics, job ids, wandb, tags
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
            "Open a research question — the TOP-LEVEL entity that diffs, experiments, assets and takeaways "
            "hang off. Create ONE question per research line and reuse it for everything in that line: call "
            "ledger_search FIRST and pass the existing id instead of creating a near-duplicate. A question is "
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
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "run_url": {
                                "type": "string",
                                "description": "wandb RUN url: https://wandb.ai/<entity>/<project>/runs/<run_id>",
                            },
                            "key": {"type": "string", "description": "wandb metric key, e.g. Train/mean_reward"},
                            "label": {"type": "string", "description": "short label; default: last '/' segment"},
                        },
                        "required": ["run_url", "key"],
                        "additionalProperties": False,
                    },
                    "description": (
                        "1-3 wandb metrics to poll and plot on the dashboard; pick the ones that answer the "
                        "research question, not every logged scalar"
                    ),
                },
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
            "conclusion (and status) of a question once the evidence is in. Also appends takeaways to a "
            "question (the permanent, searchable findings), relinks a diff/experiment to a question, and "
            "appends job ids, wandb runs, tags and diff links. Diff review status and notes are set by the "
            "user in the GUI and are rejected here."
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
                "add_takeaways": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "questions only: short self-contained declarative findings, one fact each, written so "
                        "an agent with zero context can act on them; appended and deduped"
                    ),
                },
                "takeaways": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "questions only: REPLACE the whole takeaways list (for slimming rewrites)",
                },
                "add_metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "run_url": {
                                "type": "string",
                                "description": "wandb RUN url: https://wandb.ai/<entity>/<project>/runs/<run_id>",
                            },
                            "key": {"type": "string", "description": "wandb metric key, e.g. Train/mean_reward"},
                            "label": {"type": "string", "description": "short label; default: last '/' segment"},
                        },
                        "required": ["run_url", "key"],
                        "additionalProperties": False,
                    },
                    "description": (
                        "experiments only: attach 1-3 wandb metrics polled every 5 minutes and plotted on the "
                        "dashboard; appended and deduped on (run_url, key)"
                    ),
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
        "name": "ledger_add_asset",
        "description": (
            "Attach an artifact to any ledger entry so it can be found again: checkpoint, video, plot, "
            "dataset/zarr, log dir, wandb run url, notes file. Checkpoints and plots belong on the experiment "
            "that produced them; dataset and paper links on the question. Deduped on location."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "question, diff or experiment id"},
                "label": {"type": "string", "description": "short human label, e.g. 'ckpt iter 6000'"},
                "location": {"type": "string", "description": "path, URL, wandb link, checkpoint, zarr, video"},
            },
            "required": ["id", "label", "location"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ledger_search",
        "description": (
            "Search the accumulated research knowledge base — past research questions, their stable "
            "takeaways, experiments, and diffs. Call before starting new work on a topic and cite what you "
            "find."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "free text; try 2-3 phrasings of the topic"},
                "kind": {"type": "string", "enum": ["question", "diff", "experiment"]},
                "limit": {"type": "integer", "minimum": 1, "description": "default 8"},
            },
            "required": ["query"],
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


def as_metrics(args: dict, key: str) -> list[dict]:
    """[{run_url, key, label?}] -> validated specs with the default label filled in."""
    value = args.get(key, None)
    if value is None:
        return []
    assert isinstance(value, list), f"{key} must be an array of {{run_url, key}} objects"
    specs = []
    for item in value:
        assert isinstance(item, dict), f"{key} must be an array of {{run_url, key}} objects"
        run_url = str(item.get("run_url") or "").strip()
        metric_key = str(item.get("key") or "").strip()
        parse_run_url(run_url)  # rejects project urls: a RUN url must contain /runs/<id>
        assert metric_key, f"{key}: each entry needs a non-empty wandb metric key"
        label = str(item.get("label") or "").strip() or metric_key.rsplit("/", 1)[-1]
        specs.append({"run_url": run_url, "key": metric_key, "label": label})
    return specs


def assert_diffs_exist(root: str, diff_ids: list[str]) -> None:
    _, diffs_dir, _, _ = subdirs(root)
    for diff_id in diff_ids:
        assert os.path.exists(os.path.join(diffs_dir, f"{diff_id}.json")), f"unknown diff id: {diff_id}"


def question_takeaways(root: str, question_id: str) -> list[str]:
    questions_dir, _, _, _ = subdirs(root)
    path = os.path.join(questions_dir, f"{question_id}.json")
    if not os.path.exists(path):
        return []
    return [t for t in (load_entry(path).get("takeaways") or []) if isinstance(t, str)]


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
        "takeaways": [],
        "tags": as_list(args, "tags"),
        "assets": [],
        "created": ts,
        "updated": ts,
    }
    write_entry(os.path.join(questions_dir, f"{entry_id}.json"), entry)
    return "\n".join([entry_id, *brevity_warnings(args), "", SEARCH_DIRECTIVE])


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
        "assets": [],
        "created": ts,
        "updated": ts,
    }
    write_entry(os.path.join(diffs_dir, f"{entry_id}.json"), entry)
    stored = "yes" if patch_rel else "no (empty diff)"
    return "\n".join(
        [entry_id, f"recorded in {repo}/{worktree} ({branch}); patch stored: {stored}; {len(changed)} files"]
        + brevity_warnings(args)
    )


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
    metrics = as_metrics(args, "metrics")

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
        "metrics": [],
        "results": "",
        "tags": as_list(args, "tags"),
        "assets": [],
        "created": ts,
        "updated": ts,
    }
    for metric in metrics:
        add_metric(entry, metric)
    write_entry(os.path.join(exp_dir, f"{entry_id}.json"), entry)

    lines = [entry_id]
    if not entry["metrics"] and status == "running":
        lines.append(f"\n{METRICS_MISSING}")
    takeaways = question_takeaways(root, question_id) if question_id else []
    if takeaways:
        lines.append("\ntakeaways already on this question:")
        lines += [f"  - {t}" for t in takeaways]
    lines.extend(brevity_warnings(args))
    lines.append(f"\n{WANDB_DIRECTIVE}")
    lines.append(f"\n{SEARCH_DIRECTIVE}")
    return "\n".join(lines)


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
    add_metrics = as_metrics(args, "add_metrics")
    if add_metrics:
        assert kind == "experiment", "add_metrics applies to experiment entries"
        for metric in add_metrics:
            add_metric(entry, metric)
    add_tags = as_list(args, "add_tags")
    if add_tags:
        append_dedup(entry, "tags", add_tags)
    add_takeaways = as_list(args, "add_takeaways")
    if add_takeaways:
        assert kind == "question", "add_takeaways applies to question entries"
        append_dedup(entry, "takeaways", add_takeaways)
    if "takeaways" in args:
        assert kind == "question", "takeaways applies to question entries"
        assert not add_takeaways, "use either add_takeaways or takeaways, not both"
        entry["takeaways"] = as_list(args, "takeaways")

    entry["updated"] = now_ts()
    write_entry(path, entry)
    warned = brevity_warnings(args)
    return ("\n".join(warned) + "\n" if warned else "") + json.dumps(entry, indent=2)


def tool_add_asset(args: dict) -> str:
    root = ledger_root()
    entry_id = as_str(args, "id", required=True)
    label = as_str(args, "label", required=True)
    location = as_str(args, "location", required=True)
    path = entry_path(root, entry_id)
    assert path is not None, f"no ledger entry with id {entry_id}"
    entry = load_entry(path)
    add_asset(entry, label, location)
    entry["updated"] = now_ts()
    write_entry(path, entry)
    listing = "\n".join(f"  {a['label']} -> {a['location']}" for a in entry["assets"])
    return f"{entry_id}: {len(entry['assets'])} asset(s)\n{listing}"


def tool_search(args: dict) -> str:
    root = ledger_root()
    query = as_str(args, "query", required=True)
    assert query.strip(), "query must not be empty"
    kind = as_str(args, "kind")
    assert kind in (None, "question", "diff", "experiment"), "kind must be 'question', 'diff' or 'experiment'"
    limit = args.get("limit", 8)
    assert isinstance(limit, int) and not isinstance(limit, bool), "limit must be an integer"
    assert limit > 0, "limit must be positive"

    entries = load_all(root)
    if kind:
        entries = [e for e in entries if e.get("kind") == kind]
    hits = search_entries(entries, query, limit)
    if not hits:
        return f"(no matches for {query!r}) — nothing recorded on this topic yet; try another phrasing."

    blocks = []
    for entry, score, snippets in hits:
        lines = [
            f"[{score}] {entry.get('id', '')}  {entry.get('kind', '')}/{entry.get('status', '')}",
            f"  {entry.get('title', '')}",
        ]
        takeaways = entry.get("takeaways") or []
        if entry.get("kind") == "question" and takeaways:
            lines.append("  takeaways:")
            lines += [f"    - {t}" for t in takeaways]
        lines += [f"  … {snippet}" for snippet in snippets]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


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
    "ledger_add_asset": tool_add_asset,
    "ledger_search": tool_search,
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
