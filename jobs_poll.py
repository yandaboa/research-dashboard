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
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$j" "$f" "$a" "$i" "$s" "$m"
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
  printf '%s\t%s\t%s\t%s\t%s\n' "$j" "$f" "$a" "$i" "$e"
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
    died = [j for j in ended.values() if (j["iteration"] or 0) < DIED_MIN_ITERS]
    return sorted(died, key=lambda j: j.get("end", ""), reverse=True)


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
    with ThreadPoolExecutor(max_workers=len(names) + 1) as pool:
        futures = {n: pool.submit(collect_slurm, n) for n in names}
        futures["local"] = pool.submit(collect_local)
        clusters = {}
        for n, fut in futures.items():
            try:
                clusters[n] = fut.result()
            except Exception as exc:  # noqa: BLE001 -- a collector bug must not take the dashboard down
                clusters[n] = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "jobs": []}
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
