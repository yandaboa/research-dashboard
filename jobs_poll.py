"""Collect ongoing/queued jobs across Tillicum, Hyak, Delta and this machine (used by ledger_serve.py /jobs)."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ledger import atomic_write, ledger_root  # noqa: E402

CLUSTERS = {
    "tillicum": {"host": "tillicum", "user": "yandabao", "logs": "/gpfs/projects/stf/yandabao/slurm_logs"},
    "hyak": {"host": "klone-login", "user": "yandabao", "logs": "/gscratch/weirdlab/yanda/slurm_logs"},
    "delta": {"host": "delta", "user": "bao3", "logs": "/work/nvme/bipn/bao3/slurm_logs"},
}
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
SQUEUE_FMT = "%i|%P|%T|%M|%l|%D|%N|%R|%b|%j"
LOCAL_PATTERN = re.compile(r"train\.py|play\.py|collect_.*\.py|probes/")
STALL_S = 900
NO_ITER_S = 20 * 60
DIED_WINDOW = "now-24hours"  # sacct window for the "ended without training" strip
DIED_MIN_ITERS = 2
SACCT_FMT = "JobID,JobName%60,State,Elapsed,ExitCode,End"

# Remote per-job extraction: "jobid<TAB>logpath<TAB>args-line<TAB>iter-line<TAB>success-line<TAB>mtime".
# Only the tail is scanned for iteration/success (logs run to ~300k lines); the args line is near the top.
_REMOTE_LOOP = r"""
for j in {jobs}; do
  f=$(ls -t {logs}/*"$j"*.out 2>/dev/null | head -1)
  [ -z "$f" ] && continue
  a=$(grep -m1 -a "Parsed Script CLI Args" "$f" 2>/dev/null | tr -d '\t')
  i=$(tail -n 20000 "$f" 2>/dev/null | grep -a "Learning iteration" | tail -1 | tr -d '\t')
  s=$(tail -n 20000 "$f" 2>/dev/null | grep -a "Curriculum/pomdps/mean_success_rate" | tail -1 | tr -d '\t')
  [ -z "$s" ] && s=$(tail -n 20000 "$f" 2>/dev/null | grep -a "Metrics/task_command/any_step_success_rate" | tail -1 | tr -d '\t')
  m=$(stat -c %Y "$f" 2>/dev/null)
  w=$(grep -a -o "View run at https://wandb.ai/[^ ]*" "${{f%.out}}.err" "$f" 2>/dev/null | tail -1 | sed 's/.*View run at //')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$j" "$f" "$a" "$i" "$s" "$m" "$w"
done
"""


def _ssh(host: str, script: str, timeout: int = 40) -> str:
    cmd = ["ssh", *SSH_OPTS, host, "export LC_ALL=C; " + script]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 and not proc.stdout.strip():
        err = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(err[-1] if err else f"ssh exit {proc.returncode}")
    return proc.stdout


def _elapsed_s(text: str) -> int | None:
    """SLURM elapsed/limit `D-HH:MM:SS`, `HH:MM:SS`, `MM:SS`; ps etime `[[DD-]HH:]MM:SS`."""
    if not text or text in ("UNLIMITED", "NOT_SET"):
        return None
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + s


def _parse_args_line(line: str) -> tuple[str, str]:
    """`--task X` / `--task=X` and `--run_name` from a `Parsed Script CLI Args:` line."""
    task = run_name = ""
    m = re.search(r"--task[= ]+(\S+)", line)
    if m:
        task = m.group(1)
    m = re.search(r"--run_name[= ]+(\S+)", line)
    if m:
        run_name = m.group(1)
    return task, run_name


def _badge(job: dict) -> str:
    state = job["state"]
    if state == "PENDING":
        return job.get("reason") or "PENDING"
    if state == "COMPLETING":
        return "COMPLETING"
    if state == "RUNNING":
        age = job.get("log_age_s")
        if job.get("iteration") is not None and age is not None and age > STALL_S:
            return "STALLED"
        el = _elapsed_s(job.get("elapsed") or "")
        if job.get("iteration") is None and el is not None and el > NO_ITER_S:
            return "NO-ITER"
    return "OK"


def _blank_job(cluster: str) -> dict:
    return {
        "cluster": cluster,
        "job_id": "",
        "state": "",
        "partition": "",
        "elapsed": "",
        "time_limit": "",
        "nodes": "",
        "node_list": "",
        "reason": "",
        "gpus": "",
        "name": "",
        "task": "",
        "run_name": "",
        "iteration": None,
        "max_iterations": None,
        "success": None,
        "log_age_s": None,
        "log_path": "",
        "wandb_url": "",
        "badge": "OK",
    }


# Ended jobs: iteration count from the .out plus the last exception line of the .err (the "why").
_REMOTE_DIED_LOOP = r"""
for j in {jobs}; do
  f=$(ls -t {logs}/*"$j"*.out 2>/dev/null | head -1)
  [ -z "$f" ] && continue
  a=$(grep -m1 -a "Parsed Script CLI Args" "$f" 2>/dev/null | tr -d '\t')
  i=$(tail -n 20000 "$f" 2>/dev/null | grep -a "Learning iteration" | tail -1 | tr -d '\t')
  e=$(grep -a -h -E "oom_kill|Out Of Memory|^[A-Za-z_.]*(Error|Exception)[A-Za-z_.]*: " "${{f%.out}}.err" "$f" 2>/dev/null | grep -v -E "omni\.|carb\.|\[Error\]|\[Warning\]|ChildFailedError" | tail -1 | sed 's/^\[[^]]*\] //' | tr -d '\t' | cut -c1-200)
  w=$(grep -a -o "View run at https://wandb.ai/[^ ]*" "${{f%.out}}.err" "$f" 2>/dev/null | tail -1 | sed 's/.*View run at //')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$j" "$f" "$a" "$i" "$e" "$w"
done
"""


def collect_died(name: str) -> list[dict]:
    """Jobs that ended in the last 24 h with < DIED_MIN_ITERS logged iterations (import crashes, boot hangs, OOM)."""
    cfg = CLUSTERS[name]
    out = _ssh(cfg["host"], f'sacct -u {cfg["user"]} -X -n -P -S {DIED_WINDOW} -o {SACCT_FMT} 2>/dev/null')
    ended: dict[str, dict] = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 6 or not parts[0]:
            continue
        state = parts[2].split()[0]  # "CANCELLED by 123" -> CANCELLED
        if state in ("RUNNING", "PENDING", "REQUEUED", "COMPLETING", "SUSPENDED", "RESIZING"):
            continue
        if state == "CANCELLED" and _elapsed_s(parts[3]) in (0, None):
            continue  # cancelled while still queued: never ran, nothing to report
        job = _blank_job(name)
        slurm_name = parts[1]
        job.update(
            job_id=parts[0],
            state=state,
            elapsed=parts[3],
            reason=parts[4],
            end=parts[5],
            name=slurm_name if not re.match(r"^(uwlab-dist|dist-training)-\d", slurm_name) else f"job {parts[0]}",
            badge="DIED",
        )
        ended[parts[0]] = job
    if not ended:
        return []
    script = _REMOTE_DIED_LOOP.format(jobs=" ".join(shlex.quote(j) for j in ended), logs=shlex.quote(cfg["logs"]))
    out = _ssh(cfg["host"], script + " 2>/dev/null", timeout=90)
    for line in out.splitlines():
        cols = line.split("\t")
        if len(cols) < 5 or cols[0] not in ended:
            continue
        job = ended[cols[0]]
        job["log_path"] = cols[1]
        job["task"], job["run_name"] = _parse_args_line(cols[2])
        if job["run_name"]:
            job["name"] = job["run_name"]
        m = re.search(r"Learning iteration\s+(\d+)\s*/\s*(\d+)", cols[3])
        if m:
            job["iteration"], job["max_iterations"] = int(m.group(1)), int(m.group(2))
        job["error"] = cols[4].strip()
        if len(cols) > 5:
            job["wandb_url"] = cols[5].strip()
    died = [j for j in ended.values() if (j["iteration"] or 0) < DIED_MIN_ITERS]
    return sorted(died, key=lambda j: j.get("end", ""), reverse=True)


# ---------------------------------------------------------------- cluster capacity / quotas
_CAP_PARTITIONS = {
    "tillicum": re.compile(r"^gpu-h200$"),
    "hyak": re.compile(r"^(gpu-(a40|a100|l40|l40s|h200)|ckpt|ckpt-all)$"),
    "delta": re.compile(r"^(ghx4|ghx4-interactive)$"),
}
# explicit list for `squeue -p` (the regex above is the authority for what we keep)
_CAP_QUEUE_PARTS = {
    "tillicum": "gpu-h200",
    "hyak": "gpu-a40,gpu-a100,gpu-l40,gpu-l40s,gpu-h200,ckpt,ckpt-all",
    "delta": "ghx4,ghx4-interactive",
}
_CAP_GPU_TYPES = {"hyak": re.compile(r"^(a40|a100|l40|l40s|h200)$")}  # ckpt lists every node type
_GRES_RE = re.compile(r"gpu:(?:([A-Za-z0-9_.]+):)?(\d+)")
_TRES_GPU_RE = re.compile(r"gres/gpu=(\d+)")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_HEALTHY_STATES = ("alloc", "mix", "idle", "comp")
_GPU_TYPE_ALIAS = {"nvidia_gh200_120gb": "gh200"}


def _capacity_script(name: str) -> str:
    parts = [
        'echo "== gres"',
        'sinfo -h -N -O "NodeList:30,Partition:30,StateCompact:20,Gres:60,GresUsed:80" 2>/dev/null',
        'echo "== queue"',
        f'squeue -h -p {_CAP_QUEUE_PARTS[name]} -O "UserName:30,StateCompact:10,Partition:40,tres-alloc:120" 2>/dev/null',
        'echo "== quota"',
    ]
    if name == "hyak":
        parts.append("hyakalloc 2>/dev/null")
    elif name == "tillicum":
        parts += ["hyakusage -p 2>/dev/null", 'echo "== usage"', "hyakusage 2>/dev/null"]
    elif name == "delta":
        parts.append("accounts 2>/dev/null")
    return "; ".join(parts)


def _sections(out: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    cur = "pre"
    for line in out.splitlines():
        m = re.match(r"^== (\w+)$", line.strip())
        if m:
            cur = m.group(1)
            sections[cur] = []
            continue
        sections.setdefault(cur, []).append(line)
    return sections


def _gres_gpus(field: str) -> tuple[str, int] | None:
    m = _GRES_RE.search(field)
    if not m:
        return None
    gtype = m.group(1) or "gpu"
    return _GPU_TYPE_ALIAS.get(gtype, gtype), int(m.group(2))


def _parse_gres(lines: list[str], name: str) -> list[dict]:
    keep = _CAP_PARTITIONS[name]
    types = _CAP_GPU_TYPES.get(name)
    agg: dict[tuple[str, str], dict] = {}
    for line in lines:
        cols = line.split()
        if len(cols) < 5:
            continue
        partition = cols[1].rstrip("*")
        if not keep.match(partition):
            continue
        gres, used_field = _gres_gpus(cols[3]), _gres_gpus(cols[4])
        if gres is None:
            continue
        gtype, total = gres
        if types and not types.match(gtype):
            continue
        used = used_field[1] if used_field else 0
        state = cols[2].rstrip("*-+")
        row = agg.setdefault(
            (partition, gtype),
            {"partition": partition, "type": gtype, "total": 0, "used": 0, "free": 0, "down": 0,
             "nodes": 0, "nodes_down": 0},
        )
        row["nodes"] += 1
        if state.startswith(_HEALTHY_STATES):
            row["total"] += total
            row["used"] += used
        else:
            row["down"] += total
            row["nodes_down"] += 1
    for row in agg.values():
        row["free"] = row["total"] - row["used"]
    return sorted(agg.values(), key=lambda r: (r["partition"], r["type"]))


def _parse_queue(lines: list[str], name: str) -> list[dict]:
    keep = _CAP_PARTITIONS[name]
    me = CLUSTERS[name]["user"]
    agg: dict[str, dict] = {}
    for line in lines:
        cols = line.split()
        if len(cols) < 3:
            continue
        user, state, partitions = cols[0], cols[1], cols[2]
        partition = next((p.rstrip("*") for p in partitions.split(",") if keep.match(p.rstrip("*"))), None)
        if partition is None:
            continue
        tres = cols[3] if len(cols) > 3 else ""
        m = _TRES_GPU_RE.search(tres)
        gpus = int(m.group(1)) if m else 0
        row = agg.setdefault(
            partition,
            {"partition": partition, "running_jobs": 0, "running_gpus": 0, "pending_jobs": 0,
             "pending_gpus": 0, "pending_others_jobs": 0, "pending_others_gpus": 0},
        )
        if state == "R":
            row["running_jobs"] += 1
            row["running_gpus"] += gpus
        elif state == "PD":
            row["pending_jobs"] += 1
            row["pending_gpus"] += gpus
            if user != me:
                row["pending_others_jobs"] += 1
                row["pending_others_gpus"] += gpus
    return sorted(agg.values(), key=lambda r: r["partition"])


def _parse_hyakalloc(lines: list[str]) -> list[dict]:
    """hyakalloc's box table: account+partition appear only on the TOTAL row of each block."""
    rows: dict[tuple[str, str], dict] = {}
    account = partition = ""
    for line in lines:
        if not re.search(r"[│|]", line):
            continue
        cells = [c.strip() for c in re.split(r"[│|]", _ANSI_RE.sub("", line))]
        cells = [c for c in cells if c != ""] or []
        if len(cells) < 4:
            continue
        kind = cells[-1]
        if kind not in ("TOTAL", "USED", "FREE"):
            continue
        nums = cells[-4:-1]  # CPUS, MEMORY, GPUS
        try:
            gpus = int(nums[2])
        except ValueError:
            continue
        if kind == "TOTAL":
            if len(cells) < 6:
                continue
            account, partition = cells[0], cells[1]
            rows[(account, partition)] = {"account": account, "partition": partition,
                                          "total": gpus, "used": 0, "free": 0}
        elif (account, partition) in rows:
            rows[(account, partition)]["used" if kind == "USED" else "free"] = gpus
    return sorted((r for r in rows.values() if r["total"] > 0), key=lambda r: -r["free"])


def _parse_tillicum_quota(csv_lines: list[str], usage_lines: list[str], user: str) -> dict | None:
    quota = None
    for line in csv_lines:
        cols = _ANSI_RE.sub("", line).strip().split(",")
        if len(cols) < 8 or cols[0] in ("account", ""):
            continue
        try:
            total, used = float(cols[5]), float(cols[6])
        except ValueError:
            continue
        quota = {
            "unit": "USD",
            "used": used,
            "total": total,
            "period": f"{cols[2]} → {cols[3]}",
            "label": f"{cols[0]} {cols[4]} budget",
            "gpu_hours": None,
            "mine": None,
            "mine_label": user,
        }
        break
    if quota is None:
        return None
    for line in usage_lines:
        clean = _ANSI_RE.sub("", line)
        m = re.search(r"TOTAL Usage:\s*([\d.]+)\s*GPU hours", clean)
        if m:
            quota["gpu_hours"] = float(m.group(1))
        if user in clean and quota["mine"] is None:
            m = re.search(r"\$([\d.]+)", clean)
            if m:
                quota["mine"] = float(m.group(1))
    return quota


def _parse_delta_quota(lines: list[str]) -> dict | None:
    for line in lines:
        cols = _ANSI_RE.sub("", line).split()
        if len(cols) < 3 or not cols[1].isdigit() or not cols[2].isdigit():
            continue
        balance, deposited = int(cols[1]), int(cols[2])
        return {
            "unit": "GPU h",
            "used": deposited - balance,
            "total": deposited,
            "period": None,
            "label": f"{cols[0]} allocation",
        }
    return None


def collect_capacity(name: str) -> dict:
    """Per-partition GPU availability, queue pressure and the account quota for one cluster."""
    cfg = CLUSTERS[name]
    result = {"ok": True, "error": None, "gpus": [], "queue": [], "accounts": [], "quota": None}
    try:
        out = _ssh(cfg["host"], _capacity_script(name), timeout=60)
        sec = _sections(out)
        result["gpus"] = _parse_gres(sec.get("gres", []), name)
        result["queue"] = _parse_queue(sec.get("queue", []), name)
        if name == "hyak":
            result["accounts"] = _parse_hyakalloc(sec.get("quota", []))
        elif name == "tillicum":
            result["quota"] = _parse_tillicum_quota(sec.get("quota", []), sec.get("usage", []), cfg["user"])
        elif name == "delta":
            result["quota"] = _parse_delta_quota(sec.get("quota", []))
    except Exception as exc:  # noqa: BLE001 -- capacity is decoration; never fail the jobs list
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "gpus": [], "queue": [],
                "accounts": [], "quota": None}
    return result


def collect_slurm(name: str) -> dict:
    cfg = CLUSTERS[name]
    result = {"ok": True, "error": None, "jobs": []}
    try:
        out = _ssh(cfg["host"], f'squeue -u {cfg["user"]} -h -o "{SQUEUE_FMT}" 2>/dev/null')
    except (subprocess.TimeoutExpired, RuntimeError, OSError) as exc:
        return {"ok": False, "error": str(exc) or type(exc).__name__, "jobs": []}

    jobs: dict[str, dict] = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 9 or not parts[0]:
            continue
        job = _blank_job(name)
        job.update(
            job_id=parts[0],
            partition=parts[1],
            state=parts[2],
            elapsed=parts[3],
            time_limit=parts[4],
            nodes=parts[5],
            node_list=parts[6],
            reason="" if parts[2] == "RUNNING" or parts[7] in ("None", "") else parts[7].strip("()"),
            gpus=parts[8].replace("gres/gpu:", "").replace("gres/gpu", ""),
        )
        # SLURM job name (the submit scripts set it to --run_name; older jobs carry a timestamp name)
        slurm_name = parts[9] if len(parts) > 9 else ""
        job["name"] = slurm_name if slurm_name and not re.match(r"^(uwlab-dist|dist-training)-\d", slurm_name) else f"job {parts[0]}"
        jobs[parts[0]] = job

    if jobs:
        script = _REMOTE_LOOP.format(jobs=" ".join(shlex.quote(j) for j in jobs), logs=shlex.quote(cfg["logs"]))
        try:
            out = _ssh(cfg["host"], script + " 2>/dev/null", timeout=90)
        except (subprocess.TimeoutExpired, RuntimeError, OSError) as exc:
            result["error"] = f"log scan failed: {exc}"
            out = ""
        now = time.time()
        for line in out.splitlines():
            cols = line.split("\t")
            if len(cols) < 6 or cols[0] not in jobs:
                continue
            job = jobs[cols[0]]
            job["log_path"] = cols[1]
            job["task"], job["run_name"] = _parse_args_line(cols[2])
            if job["run_name"]:
                job["name"] = job["run_name"]
            elif job["task"]:
                job["name"] = job["task"]
            m = re.search(r"Learning iteration\s+(\d+)\s*/\s*(\d+)", cols[3])
            if m:
                job["iteration"], job["max_iterations"] = int(m.group(1)), int(m.group(2))
            m = re.search(r"success_rate:\s*([-+0-9.eE]+)", cols[4])
            if m:
                try:
                    job["success"] = float(m.group(1))
                except ValueError:
                    pass
            if cols[5].strip().isdigit():
                job["log_age_s"] = int(now - int(cols[5]))
            if len(cols) > 6:
                job["wandb_url"] = cols[6].strip()

    for job in jobs.values():
        job["badge"] = _badge(job)
    result["jobs"] = sorted(jobs.values(), key=lambda j: (j["state"] != "RUNNING", j["job_id"]))
    try:
        result["died"] = collect_died(name)
    except (subprocess.TimeoutExpired, RuntimeError, OSError) as exc:
        result["died"] = []
        result["error"] = (result["error"] + "; " if result["error"] else "") + f"sacct scan failed: {exc}"
    return result


def collect_local() -> dict:
    result = {"ok": True, "error": None, "jobs": [], "gpus": []}
    try:
        out = subprocess.run(["ps", "-eo", "pid,etime,pcpu,args"], capture_output=True, text=True, timeout=10).stdout
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"ok": False, "error": str(exc), "jobs": [], "gpus": []}
    for line in out.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, etime, pcpu, args = parts
        if not LOCAL_PATTERN.search(args) or re.search(r"\bgrep\b|viser", args):
            continue
        if not re.search(r"\bpython[0-9.]*\b", args) or re.match(r"(nohup |timeout \S+ )?(ba)?sh -c", args):
            continue  # shell wrappers around the python process would double-count the run
        job = _blank_job("local")
        task, run_name = _parse_args_line(args)
        script = next((tok for tok in args.split() if tok.endswith(".py")), "")
        job.update(
            job_id=pid,
            state="RUNNING",
            elapsed=etime,
            task=task,
            run_name=run_name,
            name=run_name or task or os.path.basename(script) or f"pid {pid}",
            partition=f"cpu {pcpu}%",
        )
        job["badge"] = "OK"
        result["jobs"].append(job)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        for line in out.splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) == 4:
                result["gpus"].append({"index": cols[0], "mem_used": cols[1], "mem_total": cols[2], "util": cols[3]})
    except (subprocess.TimeoutExpired, OSError) as exc:
        result["error"] = f"nvidia-smi: {exc}"
    return result


def collect() -> dict:
    names = list(CLUSTERS)
    with ThreadPoolExecutor(max_workers=2 * len(names) + 1) as pool:
        futures = {n: pool.submit(collect_slurm, n) for n in names}
        futures["local"] = pool.submit(collect_local)
        caps = {n: pool.submit(collect_capacity, n) for n in names}
        clusters = {}
        for n, fut in futures.items():
            try:
                clusters[n] = fut.result()
            except Exception as exc:  # noqa: BLE001 -- a collector bug must not take the dashboard down
                clusters[n] = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "jobs": []}
        for n, fut in caps.items():
            try:
                clusters[n]["capacity"] = fut.result()
            except Exception as exc:  # noqa: BLE001
                clusters[n]["capacity"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"generated": datetime.now().isoformat(timespec="seconds"), "clusters": clusters}


def snapshot_path(root: str | None = None) -> str:
    return os.path.join(root or ledger_root(), "jobs_live.json")


def write_snapshot(snap: dict, root: str | None = None) -> str:
    path = snapshot_path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    atomic_write(path, json.dumps(snap, indent=1) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll live jobs across clusters.")
    parser.add_argument("--loop", type=int, default=0, metavar="N", help="write data/jobs_live.json every N seconds")
    args = parser.parse_args()
    if args.loop <= 0:
        print(json.dumps(collect(), indent=1))
        return
    while True:
        t0 = time.time()
        path = write_snapshot(collect())
        print(f"{datetime.now().isoformat(timespec='seconds')} wrote {path}", flush=True)
        time.sleep(max(1.0, args.loop - (time.time() - t0)))


if __name__ == "__main__":
    main()
