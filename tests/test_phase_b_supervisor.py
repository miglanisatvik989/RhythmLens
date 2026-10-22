"""Generated metadata, mocked resource telemetry and harmless subprocesses only."""

import importlib
import json
from pathlib import Path
import signal
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
s = importlib.import_module("supervise_phase_b")


def budgets(**changes):
    values = dict(
        training_wall_seconds=4.0,
        training_cleanup_seconds=0.3,
        evaluation_wall_seconds=4.0,
        evaluation_cleanup_seconds=0.3,
        overall_wall_seconds=8.0,
        host_rss_bytes=2 * s.GIB,
        training_gpu_bytes=7 * s.GIB,
        training_output_bytes=8 * s.GIB,
        total_output_bytes=12 * s.GIB,
        min_host_available_bytes=s.GIB,
        min_disk_free_bytes=10 * s.GIB,
    )
    return s.Budgets(**{**values, **changes})


def good(_):
    return dict(
        host_available_bytes=4 * s.GIB, disk_free_bytes=100 * s.GIB, gpu_device_bytes=0
    )


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def command(root, stage, tail=""):
    state = "training_completed" if stage == "train" else "completed"
    body = f"""import pathlib,json,os
p=pathlib.Path({str(root)!r})
{"p.mkdir()" if stage == "train" else "assert json.loads((p/'status.json').read_text())['state']=='training_completed'"}
(p/{stage!r}).mkdir()
(p/'status.json').write_text(json.dumps({{'state':{state!r}}}))
(p/{stage!r}/'environment.json').write_text(json.dumps({{'cuda':os.environ.get('CUDA_VISIBLE_DEVICES')}}))
{tail}
"""
    return [sys.executable, "-c", body]


def run(tmp_path, *, cmds=None, reader=good, limits=None, **kw):
    root = tmp_path / "run"
    logs = tmp_path / "logs"
    cmds = cmds or {stage: command(root, stage) for stage in ("train", "evaluate")}

    def verify_train():
        assert s.read_json(root / "status.json")["state"] == "training_completed"

    def verify_eval():
        assert s.read_json(root / "status.json")["state"] == "completed"

    return s.supervise(
        tmp_path,
        root,
        logs,
        limits or budgets(),
        cmds,
        kw.pop("verify_train", verify_train),
        kw.pop("verify_eval", verify_eval),
        kw.pop("verify_source", lambda: None),
        reader=reader,
        poll=0.01,
        refresh=0.01,
        **kw,
    )


def test_handoff_does_not_precreate_root_and_disables_cuda_for_cpu(tmp_path):
    result = run(tmp_path)
    assert result["state"] == "completed"
    assert s.read_json(tmp_path / "run/evaluate/environment.json")["cuda"] == ""
    assert (tmp_path / "logs/train-verified.json").exists()
    assert (tmp_path / "logs/evaluate-verified.json").exists()
    assert (tmp_path / "logs/train-stdout.log").exists()
    assert (tmp_path / "logs/claim.json").exists()
    assert "device-wide" in s.render(s.read_json(tmp_path / "logs/progress.json"))
    with pytest.raises(FileExistsError):
        run(tmp_path)


@pytest.mark.parametrize("kind", ["exit", "verification", "source"])
def test_training_failure_blocks_evaluation(tmp_path, kind):
    root = tmp_path / "run"
    cmds = {
        stage: command(
            root,
            stage,
            "raise SystemExit(7)" if kind == "exit" and stage == "train" else "",
        )
        for stage in ("train", "evaluate")
    }

    def fail():
        raise ValueError("generated verification failure")

    options = {"verify_train": fail} if kind == "verification" else {}
    if kind == "source":
        options = {"verify_source": fail}
    with pytest.raises((RuntimeError, ValueError)):
        run(tmp_path, cmds=cmds, **options)
    assert not (root / "evaluate").exists()
    assert not (tmp_path / "logs/evaluate-launch.json").exists()
    assert s.read_json(tmp_path / "logs/status.json")["state"] == "failed"


@pytest.mark.parametrize(
    "name,value,stage,expected",
    [
        ("host_rss_bytes", 5 * s.GIB, "train", "RSS"),
        ("host_available_bytes", 0, "train", "RAM"),
        ("disk_free_bytes", 0, "evaluate", "disk"),
        ("gpu_device_bytes", 8 * s.GIB, "train", "GPU"),
        ("output_bytes", 9 * s.GIB, "train", "output"),
        ("output_bytes", 13 * s.GIB, "evaluate", "output"),
    ],
)
def test_each_budget_breach(name, value, stage, expected):
    vals = {**good(None), "host_rss_bytes": 0, "output_bytes": 0, name: value}
    assert expected in s.breach(vals, budgets(), stage, 0, 10)
    if name == "gpu_device_bytes":
        assert s.breach(vals, budgets(), "evaluate", 0, 10) is None


@pytest.mark.parametrize("fault", ["memory", "gpu", "disk", "exception"])
def test_telemetry_fails_closed_before_launch(tmp_path, fault):
    def broken(_):
        if fault == "exception":
            raise OSError("generated telemetry failure")
        v = good(None)
        del v[
            {
                "memory": "host_available_bytes",
                "gpu": "gpu_device_bytes",
                "disk": "disk_free_bytes",
            }[fault]
        ]
        return v

    with pytest.raises((ValueError, OSError)):
        run(tmp_path, reader=broken)
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "logs/train-launch.json").exists()
    assert s.read_json(tmp_path / "logs/status.json")["state"] == "failed"


def live_command(root, stubborn=False):
    return [
        sys.executable,
        "-c",
        f"""import pathlib,json,signal,time,subprocess,sys
p=pathlib.Path({str(root)!r});p.mkdir()
def stop(*args):
 (p/'interrupted.json').write_text('{{"retained":true}}')
 raise SystemExit(130)
signal.signal(signal.SIGINT,signal.SIG_IGN if {stubborn!r} else stop)
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
(p/'ready.json').write_text(json.dumps({{'child':child.pid}}))
while True:time.sleep(.01)
""",
    ]


@pytest.mark.parametrize("failure", ["rss", "telemetry", "output", "gpu"])
def test_active_fault_cleans_descendants_preserves_evidence(tmp_path, failure):
    root = tmp_path / "run"
    cmds = {"train": live_command(root), "evaluate": command(root, "evaluate")}

    def reader(_):
        v = good(None)
        if (root / "ready.json").exists():
            if failure == "telemetry":
                raise OSError("generated telemetry fault")
            if failure == "gpu":
                v["gpu_device_bytes"] = 8 * s.GIB
        return v

    limits = (
        budgets(host_rss_bytes=1)
        if failure == "rss"
        else budgets(training_output_bytes=1)
        if failure == "output"
        else budgets()
    )
    # Defer tiny resource limits until the child's signal handler and metadata exist.
    original = s.sample

    def sampled(*args):
        result, reason = original(*args)
        if failure in ("rss", "output") and not (root / "ready.json").exists():
            reason = None
        return result, reason

    from unittest.mock import patch

    with patch.object(s, "sample", sampled):
        with pytest.raises((RuntimeError, OSError)):
            run(tmp_path, cmds=cmds, reader=reader, limits=limits)
    assert (root / "interrupted.json").exists()
    child = s.read_json(root / "ready.json")["child"]
    assert child not in s.processes()
    assert not (tmp_path / "logs/evaluate-launch.json").exists()
    assert s.read_json(tmp_path / "logs/train-failure.json")["cleanup_complete"]


@pytest.mark.parametrize("global_first", [False, True])
def test_deadlines_and_forced_cleanup_are_bounded(tmp_path, global_first):
    root = tmp_path / "run"
    limits = budgets(
        training_wall_seconds=2 if global_first else 0.45,
        training_cleanup_seconds=0.15,
        evaluation_cleanup_seconds=0.1,
        overall_wall_seconds=0.45 if global_first else 8,
    )
    cmds = {
        "train": live_command(root, stubborn=True),
        "evaluate": command(root, "evaluate"),
    }
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="deadline"):
        run(tmp_path, cmds=cmds, limits=limits)
    assert time.monotonic() - started < 2
    failure = s.read_json(tmp_path / "logs/train-failure.json")
    assert failure["forced_termination"] and not failure["cleanup_complete"]
    assert not (tmp_path / "logs/evaluate-launch.json").exists()


def test_mock_clock_earliest_deadline_and_cleanup(tmp_path):
    values = {**good(None), "host_rss_bytes": 0, "output_bytes": 0}
    assert s.breach(values, budgets(), "train", 99.9, 100) is None
    assert (
        s.breach(values, budgets(), "train", 100, 100)
        == "stage/global cleanup deadline"
    )
    tick = [0.0]
    calls = []

    class Tree:
        def send(self, sig):
            calls.append(sig)

        def snapshot(self):
            return [123], 0

    class Child:
        def poll(self):
            return None

    def sleep(n):
        tick[0] += n

    result = s.cleanup(Child(), Tree(), 0.3, lambda: tick[0], sleep, 0.1)
    assert tick[0] == pytest.approx(0.3)
    assert calls == [signal.SIGINT, signal.SIGKILL] and result["forced_termination"]


def test_pid_reuse_and_unrelated_processes_never_signaled(monkeypatch):
    current = {
        10: dict(start="1", session=10, ppid=0, rss=1),
        11: dict(start="2", session=10, ppid=10, rss=2),
        99: dict(start="3", session=99, ppid=0, rss=100),
    }
    monkeypatch.setattr(s, "identity", lambda pid: current[pid]["start"])
    monkeypatch.setattr(s, "processes", lambda: current)
    monkeypatch.setattr(s.os, "pidfd_open", lambda pid: pid + 1000)
    monkeypatch.setattr(s.os, "close", lambda fd: None)
    signaled = []
    monkeypatch.setattr(
        s.signal, "pidfd_send_signal", lambda fd, sig: signaled.append(fd)
    )
    tree = s.OwnedTree(10)
    assert tree.snapshot() == ([10, 11], 3)
    # A different process now owns root PID/session 10 and has its own child.
    current = {
        10: dict(start="50", session=10, ppid=0, rss=8),
        12: dict(start="51", session=10, ppid=10, rss=9),
        11: dict(start="2", session=10, ppid=1, rss=2),
        99: dict(start="3", session=99, ppid=0, rss=100),
    }
    assert tree.snapshot() == ([11], 2)
    tree.send(signal.SIGINT)
    assert signaled == [
        1010,
        1011,
    ]  # Stable handles still refer to old root/child, not new PID 10.
    assert 1099 not in signaled and 1012 not in signaled
    tree.close()


def test_reparented_and_new_session_descendants_retained(monkeypatch):
    current = {
        10: dict(start="1", session=10, ppid=0, rss=1),
        11: dict(start="2", session=11, ppid=10, rss=2),
    }
    monkeypatch.setattr(s, "identity", lambda pid: current[pid]["start"])
    monkeypatch.setattr(s, "processes", lambda: current)
    monkeypatch.setattr(s.os, "pidfd_open", lambda pid: pid)
    monkeypatch.setattr(s.os, "close", lambda fd: None)
    tree = s.OwnedTree(10)
    assert tree.snapshot() == ([10, 11], 3)
    del current[10]
    current[11]["ppid"] = 1
    assert tree.snapshot() == ([11], 2)
    tree.close()


def test_output_accounting_progress_rendering_and_no_array_reads(tmp_path):
    root = tmp_path / "root"
    logs = tmp_path / "logs"
    logs.mkdir()
    put(
        root / "progress.json",
        dict(
            stage="training",
            seed=43,
            epoch=4,
            completed=3,
            total=10,
            remaining_stage_seconds=14,
        ),
    )
    (root / "opaque.npy").write_bytes(b"never parsed")
    (logs / "runner.log").write_bytes(b"log")
    expected = sum(
        p.stat().st_size for r in (root, logs) for p in r.rglob("*") if p.is_file()
    )
    assert s.output_bytes(root, logs) == expected
    meta = s.progress(root)
    assert meta["remaining_stage_seconds"] == 14
    snapshot = {
        **good(None),
        "host_rss_bytes": 0,
        "output_bytes": expected,
        "elapsed_seconds": 4.0,
        "stage": "train",
        "runner_progress": meta,
    }
    text = s.render(snapshot)
    assert (
        "epoch=4" in text and "3/10" in text and "current epoch/condition only" in text
    )
    put(root / "progress.json", {**meta, "stage": "bootstrap_scoring"})
    assert s.progress(root)["total"] is None
    assert s.progress(root)["remaining_stage_seconds"] is None
    snapshot["runner_progress"] = s.progress(root)
    assert "[indeterminate]" in s.render(snapshot)


@pytest.mark.parametrize("target", ["run", "logs"])
def test_existing_and_dangling_destinations_rejected(tmp_path, target):
    (tmp_path / target).symlink_to(tmp_path / "absent", target_is_directory=True)
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert not (tmp_path / "absent").exists()


def test_device_telemetry_is_device_wide_and_invalid_data_fails(tmp_path, monkeypatch):
    read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **kw: "MemAvailable: 2097152 kB\n"
        if str(p) == "/proc/meminfo"
        else read(p, *a, **kw),
    )
    monkeypatch.setattr(s.subprocess, "check_output", lambda *a, **kw: "8192, 7000\n")
    monkeypatch.setattr(
        s.shutil, "disk_usage", lambda p: type("Disk", (), {"free": 20 * s.GIB})()
    )
    values = s.telemetry(tmp_path)
    assert values["gpu_device_bytes"] == 1192 * 1024**2
    assert values["host_available_bytes"] == 2 * s.GIB
    monkeypatch.setattr(s.subprocess, "check_output", lambda *a, **kw: "N/A, N/A\n")
    with pytest.raises(ValueError):
        s.telemetry(tmp_path)


def test_fixed_commands_and_disabled_cli_does_not_probe_or_create(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    commands = s.commands_for(project)
    assert commands["train"][0].endswith("environments/cuda128/.venv/bin/python")
    assert commands["evaluate"][0].endswith(".venv/bin/python")
    monkeypatch.setattr(s, "telemetry", lambda *a: pytest.fail("real telemetry probe"))
    argv = ["supervise_phase_b.py", "--root", str(project)]
    for key, value in vars(budgets()).items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="authorization"):
        s.main()
    assert not (project / s.RUN).exists() and not (project / s.LOGS).exists()


def test_global_clock_includes_verification_and_prevents_handoff(tmp_path):
    tick = [0.0]

    def clock():
        return tick[0]

    def sleep(n):
        tick[0] += n
        time.sleep(0.002)

    def verify():
        tick[0] += 0.9

    with pytest.raises(
        (RuntimeError, TimeoutError), match="deadline|verification exceeded budget"
    ):
        run(
            tmp_path,
            limits=budgets(overall_wall_seconds=1.0),
            clock=clock,
            sleep=sleep,
            verify_train=verify,
        )
    assert not (tmp_path / "logs/evaluate-launch.json").exists()


def test_monitor_failure_on_progress_preserves_outputs_and_stops(tmp_path):
    root = tmp_path / "run"
    cmds = {"train": live_command(root), "evaluate": command(root, "evaluate")}

    def reader(_):
        if (root / "ready.json").exists():
            (root / "progress.json").write_text("{malformed")
        return good(None)

    with pytest.raises(json.JSONDecodeError):
        run(tmp_path, cmds=cmds, reader=reader)
    assert (root / "progress.json").read_text() == "{malformed"
    assert (root / "interrupted.json").exists()
    assert not (tmp_path / "logs/evaluate-launch.json").exists()


def test_cli_checks_authorized_metadata_and_both_interpreters_with_mocks(
    tmp_path, monkeypatch
):
    from rhythmlens import noise_phase_b as phase

    config = tmp_path / "configs/noise-phase-b-v1.yaml"
    put(config, {"generated": True})
    import hashlib

    hashed = hashlib.sha256(config.read_bytes()).hexdigest()
    c = {
        "execution_authorized": True,
        "metadata_sha256": {},
        "environments": {
            "train": {"torch": "generated-cuda"},
            "evaluate": {"torch": "generated-cpu"},
        },
    }
    monkeypatch.setattr(phase, "load_config", lambda path: c)
    monkeypatch.setattr(phase, "require_committed_source", lambda *a: None)
    checks = []
    monkeypatch.setattr(
        phase.phase_a, "check_identities", lambda root, hashes: checks.append(hashes)
    )

    def output(args, **kw):
        if args[:2] == ["git", "ls-files"]:
            return "configs/noise-phase-b-v1.yaml\n"
        if args[:2] == ["git", "rev-parse"]:
            return "generated-revision\n"
        stage = "train" if "cuda128" in args[0] else "evaluate"
        return json.dumps(c["environments"][stage])

    monkeypatch.setattr(s.subprocess, "check_output", output)
    monkeypatch.setattr(
        phase,
        "completed_training",
        lambda p: {
            "source_revision": "generated-revision",
            "config_byte_sha256": hashed,
        },
    )

    def supervise(
        project, root, logs, limits, commands, verify_train, verify_eval, verify_source
    ):
        verify_source()
        put(root / "status.json", {"state": "training_completed"})
        verify_train()
        put(
            root / "evaluation/status.json",
            {"state": "completed", "artifacts_index_sha256": "generated"},
        )
        for name in (
            "comparison-seal.json",
            "seal.json",
            "metrics.json",
            "bootstrap.json",
        ):
            put(root / "evaluation" / name, {})
        put(
            root / "status.json",
            {
                "state": "completed",
                "evaluation_status_sha256": phase.sha(root / "evaluation/status.json"),
            },
        )
        verify_eval()
        assert commands == s.commands_for(project)

    monkeypatch.setattr(phase, "verify_index", lambda *a, **kw: None)
    monkeypatch.setattr(s, "supervise", supervise)
    argv = ["supervise_phase_b.py", "--root", str(tmp_path), "--authorize-execution"]
    for key, value in vars(budgets()).items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    monkeypatch.setattr(sys, "argv", argv)
    s.main()
    assert {"configs/noise-phase-b-v1.yaml": hashed} in checks


def test_procfs_failure_cleanup_uses_stable_handles_only():
    tick = [0.0]
    signals = []

    class Tree:
        def snapshot(self):
            raise OSError("generated procfs unavailable")

        def send(self, sig):
            signals.append(sig)

    class Child:
        def poll(self):
            return None

    def sleep(n):
        tick[0] += n

    result = s.cleanup(Child(), Tree(), 0.2, lambda: tick[0], sleep, 0.05)
    assert signals == [signal.SIGINT, signal.SIGKILL]
    assert not result["cleanup_complete"]


def test_verification_alarm_bounds_metadata_work():
    with pytest.raises(TimeoutError, match="Verification cleanup deadline"):
        s.verify_before(
            lambda: time.sleep(0.3), time.monotonic() + 0.025, time.monotonic
        )
    with pytest.raises(TimeoutError):
        s.verify_before(lambda: None, 0, lambda: 1)


def test_descendant_pid_reuse_requires_new_stable_handle(monkeypatch):
    current = {
        10: dict(start="1", session=10, ppid=0, rss=1),
        11: dict(start="2", session=10, ppid=10, rss=2),
    }
    monkeypatch.setattr(s, "identity", lambda pid: current[pid]["start"])
    monkeypatch.setattr(s, "processes", lambda: current)
    fds = iter([100, 101, 102])
    closed = []
    sent = []
    monkeypatch.setattr(s.os, "pidfd_open", lambda pid: next(fds))
    monkeypatch.setattr(s.os, "close", lambda fd: closed.append(fd))
    monkeypatch.setattr(s.signal, "pidfd_send_signal", lambda fd, sig: sent.append(fd))
    tree = s.OwnedTree(10)
    tree.snapshot()
    current[11]["start"] = (
        "3"  # new genuine child, proven through the still-owned parent/session
    )
    assert tree.snapshot() == ([10, 11], 3)
    tree.send(signal.SIGINT)
    assert closed == [101] and sent == [100, 102]
    tree.close()


def test_failed_evaluation_is_not_success_or_retried(tmp_path):
    root = tmp_path / "run"
    cmds = {
        stage: command(
            root, stage, "raise SystemExit(9)" if stage == "evaluate" else ""
        )
        for stage in ("train", "evaluate")
    }
    with pytest.raises(RuntimeError, match="evaluate exit status 9"):
        run(tmp_path, cmds=cmds)
    assert (tmp_path / "logs/train-verified.json").exists()
    assert not (tmp_path / "logs/evaluate-verified.json").exists()
    with pytest.raises(FileExistsError):
        run(tmp_path)


@pytest.mark.parametrize(
    "changes",
    [
        {"training_wall_seconds": float("inf")},
        {"evaluation_cleanup_seconds": 4.0},
        {"host_rss_bytes": 0},
        {"training_output_bytes": 13 * s.GIB},
    ],
)
def test_invalid_budgets_are_not_execution_authority(changes):
    with pytest.raises(ValueError):
        budgets(**changes)
