"""One-attempt external Phase B watchdog. Budgets never grant execution authority."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

# Reuse metadata/process primitives, never the Phase A launcher or stage monitor.
from supervise_phase_a import processes, output_bytes, write_json

GIB = 1024**3
RUN = "outputs/noise-robustness/noise-robustness-v1/phase-b"
LOGS = "outputs/noise-phase-b-supervision-v1"


@dataclass(frozen=True)
class Budgets:
    training_wall_seconds: float
    training_cleanup_seconds: float
    evaluation_wall_seconds: float
    evaluation_cleanup_seconds: float
    overall_wall_seconds: float
    host_rss_bytes: int
    training_gpu_bytes: int
    training_output_bytes: int
    total_output_bytes: int
    min_host_available_bytes: int
    min_disk_free_bytes: int

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Every budget must be finite and positive")
        if (
            self.training_cleanup_seconds >= self.training_wall_seconds
            or self.evaluation_cleanup_seconds >= self.evaluation_wall_seconds
            or max(self.training_cleanup_seconds, self.evaluation_cleanup_seconds)
            >= self.overall_wall_seconds
            or self.training_output_bytes > self.total_output_bytes
        ):
            raise ValueError("Inconsistent cleanup/output budgets")


def read_json(path):
    return json.loads(path.read_text())


def identity(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


class OwnedTree:
    """Private-session descendants plus remembered identities; pidfds avoid kill races."""

    def __init__(self, pid):
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("Linux pidfd support is required")
        self.pid, self.start = pid, identity(pid)
        self.handles = {}
        self.bind(pid, self.start)

    def bind(self, pid, start):
        if pid in self.handles and self.handles[pid][0] == start:
            return True
        try:
            fd = os.pidfd_open(pid)
            if identity(pid) != start:
                os.close(fd)
                return False
            if pid in self.handles:
                os.close(self.handles[pid][1])
            self.handles[pid] = (start, fd)
            return True
        except ProcessLookupError:
            return False
        except FileNotFoundError:
            if "fd" in locals():
                os.close(fd)
            return False

    def snapshot(self):
        current = processes()
        selected = {
            p
            for p, (start, _) in self.handles.items()
            if p in current and current[p]["start"] == start
        }
        # Never adopt the session of a recycled root PID.
        root_reused = self.pid in current and current[self.pid]["start"] != self.start
        if not root_reused:
            selected.update(
                p
                for p, v in current.items()
                if v["session"] == self.pid and int(v["start"]) >= int(self.start)
            )
        while True:
            children = {
                p
                for p, v in current.items()
                if v["ppid"] in selected and not (root_reused and p == self.pid)
            }
            if children <= selected:
                break
            selected.update(children)
        selected = {p for p in selected if self.bind(p, current[p]["start"])}
        return sorted(selected), sum(current[p]["rss"] for p in selected)

    def send(self, sig):
        for _, fd in self.handles.values():
            try:
                signal.pidfd_send_signal(fd, sig)
            except ProcessLookupError:
                pass

    def close(self):
        for _, fd in self.handles.values():
            os.close(fd)


def telemetry(project):
    memory = {
        k: int(v.split()[0]) * 1024
        for k, v in (
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
        if k == "MemAvailable"
    }
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=2,
    )
    lines = result.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("GPU 0 telemetry missing/ambiguous")
    total, free = (int(v.strip()) * 1024**2 for v in lines[0].split(","))
    if not 0 <= free <= total or total <= 0:
        raise ValueError("Invalid GPU 0 telemetry")
    return dict(
        host_available_bytes=memory["MemAvailable"],
        disk_free_bytes=shutil.disk_usage(project).free,
        gpu_device_bytes=total - free,
        gpu_total_bytes=total,
        gpu_free_bytes=free,
    )


def validate_telemetry(values):
    for key in ("host_available_bytes", "disk_free_bytes", "gpu_device_bytes"):
        if type(values.get(key)) is not int or values[key] < 0:
            raise ValueError(f"Required telemetry unavailable: {key}")


def breach(values, budgets, stage, now, interrupt_at):
    if now >= interrupt_at:
        return "stage/global cleanup deadline"
    if values["host_rss_bytes"] > budgets.host_rss_bytes:
        return "aggregate host RSS limit"
    if values["host_available_bytes"] < budgets.min_host_available_bytes:
        return "available host RAM floor"
    if values["disk_free_bytes"] < budgets.min_disk_free_bytes:
        return "free disk floor"
    if stage == "train" and values["gpu_device_bytes"] > budgets.training_gpu_bytes:
        return "GPU 0 device-wide memory limit"
    output_limit = (
        budgets.training_output_bytes
        if stage == "train"
        else budgets.total_output_bytes
    )
    if values["output_bytes"] > output_limit:
        return "generated output/log limit"
    return None


def progress(root):
    path = root / "progress.json"
    if not path.exists():
        return dict(
            stage="preflight",
            seed=None,
            epoch=None,
            completed=0,
            total=None,
            remaining_stage_seconds=None,
        )
    value = read_json(
        path
    )  # Atomic runner metadata; malformed publication is a monitor fault.
    if value["stage"] in ("clean_validation", "bootstrap_scoring", "sealing"):
        value = {**value, "total": None, "remaining_stage_seconds": None}
    total, done = value.get("total"), value.get("completed", 0)
    if (
        type(done) is not int
        or done < 0
        or (
            total is not None and (type(total) is not int or total <= 0 or done > total)
        )
    ):
        raise ValueError("Invalid published progress metadata")
    eta = value.get("remaining_stage_seconds")
    if eta is not None and (
        not isinstance(eta, (int, float)) or not math.isfinite(eta) or eta < 0
    ):
        raise ValueError("Invalid stage ETA")
    return value


def render(s):
    p = s["runner_progress"]
    total = p.get("total")
    done = p.get("completed", 0)
    bar = (
        ("[" + ("#" * int(20 * done / total)).ljust(20, ".") + "]")
        if total
        else "[indeterminate]"
    )
    eta = p.get("remaining_stage_seconds")
    return (
        f"{s['stage']} / {p['stage']} seed={p.get('seed')} epoch={p.get('epoch')} "
        f"{bar} {done}/{total or '?'} batches (current epoch/condition only) | "
        f"elapsed={s['elapsed_seconds']:.1f}s stage ETA={f'{eta:.1f}s' if eta is not None else '?'} | "
        f"host RSS={s['host_rss_bytes'] / GIB:.2f}GiB available={s['host_available_bytes'] / GIB:.2f}GiB | "
        f"GPU0 device-wide(total-free)={s['gpu_device_bytes'] / GIB:.2f}GiB | "
        f"output+logs={s['output_bytes'] / GIB:.3f}GiB"
    )


def sample(
    project, root, logs, stage, budgets, global_start, interrupt_at, tree, reader, clock
):
    owned, rss = tree.snapshot() if tree else ([], 0)
    values = reader(project)
    validate_telemetry(values)
    values.update(
        host_rss_bytes=rss,
        output_bytes=output_bytes(root, logs),
        processes=owned,
        stage=stage,
        elapsed_seconds=clock() - global_start,
        runner_progress=progress(root),
    )
    return values, breach(values, budgets, stage, clock(), interrupt_at)


def publish(logs, snapshot):
    write_json(logs / "progress.json", snapshot)
    with (logs / "snapshots.jsonl").open("a") as stream:
        stream.write(json.dumps(snapshot, allow_nan=False) + "\n")
    with (logs / "progress.log").open("a") as stream:
        stream.write(render(snapshot) + "\n")
    print(render(snapshot), flush=True)


def cleanup(runner, tree, end, clock, sleep, poll):
    # A second operator interrupt must not abandon the already-bounded cleanup.
    previous = {
        sig: signal.signal(sig, signal.SIG_IGN)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        tree.send(signal.SIGINT)
        forced = False
        observed = True
        while clock() < end:
            runner.poll()
            try:
                owned, _ = tree.snapshot()
                if not owned:
                    break
            except BaseException:
                observed = (
                    False  # Stable pidfds remain safe even when procfs is unavailable.
                )
            sleep(min(poll, max(0, end - clock())))
        else:
            forced = True
        if forced or not observed:
            tree.send(signal.SIGKILL)
        runner.poll()
        return dict(
            forced_termination=forced or not observed,
            cleanup_complete=not forced and observed,
        )
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def stage_run(
    command,
    project,
    root,
    logs,
    stage,
    budgets,
    global_start,
    reader,
    clock,
    sleep,
    poll,
    refresh,
):
    start = clock()
    wall = (
        budgets.training_wall_seconds
        if stage == "train"
        else budgets.evaluation_wall_seconds
    )
    allowance = (
        budgets.training_cleanup_seconds
        if stage == "train"
        else budgets.evaluation_cleanup_seconds
    )
    hard = min(start + wall, global_start + budgets.overall_wall_seconds)
    interrupt_at = hard - allowance
    # Before launch too: unavailable telemetry, memory/disk pressure, deadlines fail closed.
    initial, reason = sample(
        project,
        root,
        logs,
        stage,
        budgets,
        global_start,
        interrupt_at,
        None,
        reader,
        clock,
    )
    publish(logs, initial)
    if reason:
        raise RuntimeError(reason)
    tree = None
    runner = None
    try:
        with (logs / f"{stage}-stdout.log").open("xb") as output:
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            if stage == "evaluate":
                env["CUDA_VISIBLE_DEVICES"] = ""
            runner = subprocess.Popen(
                command,
                cwd=project,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            tree = OwnedTree(runner.pid)
            write_json(
                logs / f"{stage}-launch.json",
                dict(
                    command=command,
                    pid=runner.pid,
                    creation_identity=tree.start,
                    hard_deadline_monotonic=hard,
                    interrupt_deadline_monotonic=interrupt_at,
                    cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
                ),
            )
            last = -float("inf")
            while True:
                s, reason = sample(
                    project,
                    root,
                    logs,
                    stage,
                    budgets,
                    global_start,
                    interrupt_at,
                    tree,
                    reader,
                    clock,
                )
                code = runner.poll()
                if reason:
                    raise RuntimeError(reason)
                if code is not None and code != 0:
                    raise RuntimeError(f"{stage} exit status {code}")
                if clock() - last >= refresh or (
                    code is not None and not s["processes"]
                ):
                    publish(logs, s)
                    last = clock()
                if code is not None and not s["processes"]:
                    return dict(
                        exit_status=code,
                        forced_termination=False,
                        cleanup_complete=True,
                        stage_seconds=clock() - start,
                        hard_deadline_monotonic=hard,
                    )
                sleep(min(poll, max(0, interrupt_at - clock())))
    except BaseException as exc:
        result = dict(error=str(exc), stage=stage)
        if runner is not None and tree is not None:
            result.update(
                cleanup(
                    runner, tree, min(hard, clock() + allowance), clock, sleep, poll
                )
            )
        elif runner is not None:
            # Popen owns this unreaped direct child; never signal by a recycled bare PID.
            runner.send_signal(signal.SIGINT)
            try:
                runner.wait(timeout=max(0, min(hard, clock() + allowance) - clock()))
            except subprocess.TimeoutExpired:
                runner.kill()
                result.update(forced_termination=True, cleanup_complete=False)
        try:
            write_json(logs / f"{stage}-failure.json", result)
        except OSError:
            pass  # Storage failure cannot undo process cleanup.
        raise
    finally:
        if tree is not None:
            tree.close()


def verify_before(callback, end, clock):
    """Bound metadata-only handoff verification while no runner tree is alive."""
    remaining = end - clock()
    if remaining <= 0:
        raise TimeoutError("Verification cleanup deadline reached")
    old_handler = signal.getsignal(signal.SIGALRM)

    def expired(*_):
        raise TimeoutError("Verification cleanup deadline reached")

    signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        callback()
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old_handler)


def supervise(
    project,
    root,
    logs,
    budgets,
    commands,
    verify_train,
    verify_eval,
    verify_source,
    *,
    reader=telemetry,
    clock=time.monotonic,
    sleep=time.sleep,
    poll=1.0,
    refresh=5.0,
):
    if (
        not math.isfinite(poll)
        or not math.isfinite(refresh)
        or poll <= 0
        or refresh <= 0
    ):
        raise ValueError("Positive finite sampling intervals required")
    if any(p.exists() or p.is_symlink() for p in (root, logs)):
        raise FileExistsError(
            "Existing runner/supervisor destination; preserve, never retry"
        )
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Linux pidfd support required before launch")
    fd = os.pidfd_open(os.getpid())  # Check kernel support before launch.
    os.close(fd)
    logs.mkdir(
        parents=True, exist_ok=False
    )  # Exclusive supervisor claim, not runner root.
    global_start = clock()
    write_json(
        logs / "claim.json",
        dict(
            budgets=asdict(budgets),
            commands=commands,
            global_start_monotonic=global_start,
            poll_seconds=poll,
            refresh_seconds=refresh,
        ),
    )
    write_json(logs / "status.json", dict(state="running", stage="train"))
    try:
        results = {}
        for stage, verify in (("train", verify_train), ("evaluate", verify_eval)):
            # Do not reset the global clock at handoff; verification time is included.
            allowance = (
                budgets.training_cleanup_seconds
                if stage == "train"
                else budgets.evaluation_cleanup_seconds
            )
            verify_before(
                verify_source,
                global_start + budgets.overall_wall_seconds - allowance,
                clock,
            )
            write_json(logs / "status.json", dict(state="running", stage=stage))
            results[stage] = stage_run(
                commands[stage],
                project,
                root,
                logs,
                stage,
                budgets,
                global_start,
                reader,
                clock,
                sleep,
                poll,
                refresh,
            )
            hard = results[stage]["hard_deadline_monotonic"]
            verify_before(verify, hard - allowance, clock)
            verify_before(verify_source, hard - allowance, clock)
            s, reason = sample(
                project,
                root,
                logs,
                stage,
                budgets,
                global_start,
                hard - allowance,
                None,
                reader,
                clock,
            )
            publish(logs, s)
            if reason:
                raise RuntimeError(f"Completion verification exceeded budget: {reason}")
            write_json(logs / f"{stage}-verified.json", results[stage])
        result = dict(
            state="completed", elapsed_seconds=clock() - global_start, stages=results
        )
        write_json(logs / "status.json", result)
        return result
    except BaseException as exc:
        result = dict(
            state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=str(exc),
            elapsed_seconds=clock() - global_start,
        )
        try:
            write_json(logs / "status.json", result)
        except OSError:
            pass
        raise


def commands_for(project):
    common = [
        "-u",
        "-B",
        "-m",
        "rhythmlens.noise_phase_b",
        "--config",
        str(project / "configs/noise-phase-b-v1.yaml"),
        "--root",
        str(project),
        "--authorize-execution",
    ]
    return {
        stage: [str(project / interpreter), *common, "--stage", stage]
        for stage, interpreter in [
            ("train", "environments/cuda128/.venv/bin/python"),
            ("evaluate", ".venv/bin/python"),
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorize-execution", action="store_true")
    for name in Budgets.__dataclass_fields__:
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=float if name.endswith("seconds") else int,
            required=True,
        )
    args = parser.parse_args()
    project = args.root.resolve()
    # Import the existing strict parser without changing its policy or source bindings.
    import sys

    sys.path.insert(0, str(project / "src"))
    from rhythmlens import noise_phase_b as phase

    c = phase.load_config(project / "configs/noise-phase-b-v1.yaml")
    if c["execution_authorized"] is not True or args.authorize_execution is not True:
        raise ValueError(
            "Separate committed Phase B and explicit runtime authorization required"
        )
    phase.require_committed_source(project, project / "configs/noise-phase-b-v1.yaml")
    budgets = Budgets(
        **{name: getattr(args, name) for name in Budgets.__dataclass_fields__}
    )
    names = subprocess.check_output(
        ["git", "ls-files"], cwd=project, text=True
    ).splitlines()
    hashes = {n: hashlib.sha256((project / n).read_bytes()).hexdigest() for n in names}
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project, text=True
    ).strip()

    def verify_source():
        phase.require_committed_source(
            project, project / "configs/noise-phase-b-v1.yaml"
        )
        phase.phase_a.check_identities(project, hashes)
        phase.phase_a.check_identities(project, c["metadata_sha256"])
        if (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project, text=True
            ).strip()
            != revision
        ):
            raise ValueError("Execution revision changed")
        for stage, cmd in commands_for(project).items():
            code = "import json,platform,importlib.metadata as m;print(json.dumps(dict(python=platform.python_version(),**{k:m.version(k) for k in ('numpy','scipy','torch','wfdb','PyYAML')})))"
            versions = json.loads(
                subprocess.check_output(
                    [cmd[0], "-B", "-c", code], cwd=project, text=True, timeout=5
                )
            )
            if versions != c["environments"][stage]:
                raise ValueError(f"{stage} environment mismatch")

    root = project / RUN

    def verify_train():
        if read_json(root / "status.json")["state"] != "training_completed":
            raise ValueError("Training lifecycle did not complete")
        registry = phase.completed_training(root / "training")
        if (
            registry["source_revision"] != revision
            or registry["config_byte_sha256"] != hashes["configs/noise-phase-b-v1.yaml"]
        ):
            raise ValueError("Training source/configuration mismatch")

    def verify_eval():
        status = read_json(root / "status.json")
        evaluation = root / "evaluation"
        if status["state"] != "completed" or status[
            "evaluation_status_sha256"
        ] != phase.sha(evaluation / "status.json"):
            raise ValueError("Evaluation lifecycle did not complete")
        child = read_json(evaluation / "status.json")
        if child["state"] != "completed":
            raise ValueError("Evaluation failed")
        phase.verify_index(
            evaluation,
            "artifacts-sha256.json",
            child["artifacts_index_sha256"],
            exclude=("status.json",),
        )
        for name in (
            "comparison-seal.json",
            "seal.json",
            "metrics.json",
            "bootstrap.json",
        ):
            if not (evaluation / name).is_file():
                raise ValueError(f"Missing final evidence: {name}")

    verify_source()

    def interrupted(*_):
        raise KeyboardInterrupt("Supervisor received SIGTERM")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        supervise(
            project,
            root,
            project / LOGS,
            budgets,
            commands_for(project),
            verify_train,
            verify_eval,
            verify_source,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
