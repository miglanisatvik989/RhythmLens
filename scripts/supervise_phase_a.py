"""External one-attempt CPU Phase A progress/watchdog; never reads prediction arrays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

GIB = 1024**3
TOTAL_BATCHES = 13200


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def inspect_outputs(root, confirmed=None):
    """Published array+complete metadata pair, not started events, defines a batch."""
    completed = set() if confirmed is None else confirmed
    for path in root.glob("seed*/*/batch*.json"):
        if not re.fullmatch(r"batch\d+\.json", path.name):
            continue
        if path in completed:
            continue
        metadata = read_json(path)
        array = path.with_suffix(".npy")
        if metadata is None or not array.is_file():
            continue
        indices = metadata.get("indices", [])
        cases = metadata.get("cases", [])
        if (
            not indices
            or len(indices) != len(cases)
            or array.stat().st_size < 128 + 16 * len(indices)
        ):
            continue
        completed.add(path)
    done = len(completed)
    if done > TOTAL_BATCHES:
        raise ValueError("Unexpected batch inventory; stop for review")
    lifecycle = read_json(root / "status.json") or {}
    if lifecycle.get("state") in ("failed", "interrupted"):
        stage = "failure"
    elif lifecycle.get("state") == "completed":
        stage = "completion"
    elif (root / "seal.json").is_file():
        stage = "bootstrap/scoring"
    elif done == TOTAL_BATCHES:
        stage = "sealing"
    elif done or (root / "plans.json").is_file():
        gates = [
            read_json(root / f"seed{s}/clean/baseline-gate.json") for s in (42, 43, 44)
        ]
        stage = (
            "stress inference"
            if all(g and g.get("status") == "passed" for g in gates)
            else "clean reproduction"
        )
    else:
        stage = "preflight"
    return done, stage


def output_bytes(*roots):
    total = 0
    for root in roots:
        if root.exists():
            for directory, _, names in os.walk(root, followlinks=False):
                for name in names:
                    path = Path(directory) / name
                    try:
                        if not path.is_symlink():
                            total += path.stat().st_size
                    except FileNotFoundError:
                        pass  # Atomic replacement while sampling.
    return total


def processes():
    """PID identity, ancestry, session/group and resident pages from Linux procfs."""
    result = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                continue
            result[int(path.name)] = dict(
                ppid=int(fields[1]),
                group=int(fields[2]),
                session=int(fields[3]),
                start=fields[19],
                rss=int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def tree_snapshot(pid, identity, remembered):
    current = processes()
    selected = {
        p
        for p, start in remembered.items()
        if p in current and current[p]["start"] == start
    }
    if pid in current and current[pid]["start"] == identity:
        selected.add(pid)
    # A private session keeps ordinary reparented descendants attributable.
    selected.update(p for p, v in current.items() if v["session"] == pid)
    while True:
        children = {p for p, v in current.items() if v["ppid"] in selected}
        if children <= selected:
            break
        selected.update(children)
    known = {p: current[p]["start"] for p in selected}
    remembered.update(known)
    return known, sum(current[p]["rss"] for p in selected)


def signal_tree(known, sig):
    current = processes()
    for pid, identity in known.items():
        if pid in current and current[pid]["start"] == identity:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def bar(snapshot):
    done = snapshot["completed_batches"]
    filled = int(30 * done / TOTAL_BATCHES)
    eta = snapshot["remaining_inference_seconds"]
    remaining = f"{eta / 60:.1f}m" if eta is not None else "?"
    indicator = (
        " [indeterminate]"
        if snapshot["stage"] in ("bootstrap/scoring", "sealing", "preflight")
        else ""
    )
    return (
        f"{snapshot['stage']}{indicator} [{'#' * filled}{'.' * (30 - filled)}] "
        f"{done}/{TOTAL_BATCHES} {100 * done / TOTAL_BATCHES:.2f}% inference | "
        f"elapsed {snapshot['elapsed_seconds'] / 60:.1f}m | "
        f"{snapshot['batch_rate_per_second'] or 0:.2f} batch/s | inference ETA {remaining} | "
        f"RAM {snapshot['ram_bytes'] / GIB:.2f}GiB | output {snapshot['output_bytes'] / GIB:.3f}GiB"
    )


def supervise(
    command,
    root,
    logs,
    *,
    wall=28800,
    cleanup=300,
    ram_limit=4 * GIB,
    output_limit=5 * GIB,
    poll=1.0,
    refresh=5.0,
):
    if root.exists() or root.is_symlink() or logs.exists() or logs.is_symlink():
        raise FileExistsError("Existing run/log destination: preserve; no retry")
    logs.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    runner = None
    known, remembered = {}, {}
    interrupted = None
    interrupt_elapsed = None
    forced = False
    peak_ram = peak_output = 0
    first = None
    confirmed = set()
    last_display = -refresh
    stage_start, previous_stage = start, None
    with (logs / "runner.log").open("xb") as output:
        try:
            runner = subprocess.Popen(
                command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
            )
            identity = processes().get(runner.pid, {}).get("start")
            if identity is None:
                runner.wait()
            write_json(
                logs / "launch.json",
                dict(
                    pid=runner.pid,
                    process_start_identity=identity,
                    command=command,
                    started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    budgets=dict(
                        wall_seconds=wall,
                        cleanup_seconds=cleanup,
                        ram_bytes=ram_limit,
                        output_bytes=output_limit,
                    ),
                    resource_poll_seconds=poll,
                    progress_refresh_seconds=refresh,
                ),
            )
            while True:
                elapsed = time.monotonic() - start
                known, ram = tree_snapshot(runner.pid, identity, remembered)
                size = output_bytes(root, logs)
                peak_ram, peak_output = max(peak_ram, ram), max(peak_output, size)
                done, stage = inspect_outputs(root, confirmed)
                if previous_stage != stage:
                    stage_start, previous_stage = time.monotonic(), stage
                if done and first is None:
                    first = (time.monotonic(), done)
                rate = (
                    (done - first[1]) / (time.monotonic() - first[0])
                    if first
                    and done - first[1] >= 8
                    and time.monotonic() - first[0] >= 10
                    else None
                )
                snapshot = dict(
                    stage=stage,
                    completed_batches=done,
                    total_inference_batches=TOTAL_BATCHES,
                    inference_percent=100 * done / TOTAL_BATCHES,
                    elapsed_seconds=elapsed,
                    observed_stage_seconds=time.monotonic() - stage_start,
                    batch_rate_per_second=rate,
                    remaining_inference_seconds=(TOTAL_BATCHES - done) / rate
                    if rate
                    else None,
                    ram_bytes=ram,
                    output_bytes=size,
                    peak_observed_ram_bytes=peak_ram,
                    peak_observed_output_bytes=peak_output,
                    process_ids=sorted(known),
                    stop_reason=interrupted,
                    forced_termination=forced,
                )
                code = runner.poll()
                if code is not None and not known:
                    snapshot.update(
                        returncode=code,
                        stage=stage
                        if stage in ("failure", "completion")
                        else "failure",
                        cleanup_complete=not forced,
                    )
                    write_json(logs / "progress.json", snapshot)
                    print("\r" + bar(snapshot), flush=True)
                    return snapshot
                reason = (
                    "deadline"
                    if elapsed >= wall - cleanup
                    else "RAM limit"
                    if ram > ram_limit
                    else "output limit"
                    if size > output_limit
                    else None
                )
                if reason and interrupted is None:
                    interrupted = reason
                    interrupt_elapsed = elapsed
                    signal_tree(known, signal.SIGINT)
                hard_stop = (
                    min(wall, interrupt_elapsed + cleanup)
                    if interrupt_elapsed is not None
                    else wall
                )
                if elapsed >= hard_stop:
                    forced = True
                    signal_tree(known, signal.SIGKILL)
                if elapsed - last_display >= refresh:
                    write_json(logs / "progress.json", snapshot)
                    with (logs / "samples.jsonl").open("a") as stream:
                        stream.write(json.dumps(snapshot) + "\n")
                    print("\r" + bar(snapshot), end="", flush=True)
                    last_display = elapsed
                time.sleep(poll)
        except BaseException:
            if runner is not None:
                # Watchdog faults must not leave an unsupervised runner.
                known, _ = tree_snapshot(runner.pid, identity, remembered)
                signal_tree(known, signal.SIGINT)
                end = min(start + wall, time.monotonic() + cleanup)
                while known and time.monotonic() < end:
                    time.sleep(min(poll, 1))
                    runner.poll()
                    known, _ = tree_snapshot(runner.pid, identity, remembered)
                if known:
                    signal_tree(known, signal.SIGKILL)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    project = args.root.resolve()
    import yaml

    config = project / "configs/noise-phase-a-v1.yaml"
    doc = yaml.safe_load(config.read_text())
    if doc["execution_authorized"] is not True:
        raise ValueError("Commit explicit authorization before supervision")
    for name in ("noise-robustness-v1.yaml", "noise-quality-datasets-v1.yaml"):
        if (
            yaml.safe_load((project / "configs" / name).read_text())[
                "execution_authorized"
            ]
            is not False
        ):
            raise ValueError("Frozen proposal authorization changed")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=project
    ).strip():
        raise ValueError("Clean committed source required")
    if shutil.disk_usage(project / "outputs").free < 10 * GIB:
        raise ValueError("At least 10 GiB free disk required")
    result = supervise(
        [
            str(project / ".venv/bin/python"),
            "-u",
            "-B",
            "-m",
            "rhythmlens.noise_phase_a",
            "--config",
            str(config),
            "--root",
            str(project),
            "--authorize-execution",
        ],
        project / doc["output_root"],
        project / "outputs/noise-phase-a-run-v1",
    )
    if result.get("returncode") != 0 or result["stage"] != "completion":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
