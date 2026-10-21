"""External watchdog tests use only generated files and synthetic child processes."""

import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest

spec = importlib.util.spec_from_file_location(
    "phase_a_supervisor",
    Path(__file__).resolve().parents[1] / "scripts/supervise_phase_a.py",
)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_published_pairs_not_start_events_or_partial_arrays(tmp_path):
    root = tmp_path / "run"
    put(root / "status.json", {"state": "running"})
    put(root / "events.jsonl", {"event": "started"})
    assert supervisor.inspect_outputs(root) == (0, "preflight")
    path = root / "seed42/clean/batch0.json"
    put(path, {"indices": [0], "cases": [{"generated": True}]})
    assert supervisor.inspect_outputs(root)[0] == 0
    path.with_suffix(".npy").write_bytes(b"generated fixture".ljust(144, b"x"))
    assert supervisor.inspect_outputs(root) == (1, "clean reproduction")
    path.write_text("{partial")
    assert supervisor.inspect_outputs(root)[0] == 0
    assert supervisor.output_bytes(root) > 0


def test_stages_keep_inference_distinct_from_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "TOTAL_BATCHES", 1)
    root = tmp_path / "run"
    path = root / "seed42/clean/batch0.json"
    put(path, {"indices": [0], "cases": [{}]})
    path.with_suffix(".npy").write_bytes(b"x" * 144)
    assert supervisor.inspect_outputs(root)[1] == "sealing"
    put(root / "seal.json", {})
    assert supervisor.inspect_outputs(root)[1] == "bootstrap/scoring"
    put(root / "status.json", {"state": "failed"})
    assert supervisor.inspect_outputs(root)[1] == "failure"
    put(root / "status.json", {"state": "completed"})
    assert supervisor.inspect_outputs(root)[1] == "completion"


def test_success_and_collision(tmp_path):
    root = tmp_path / "run"
    logs = tmp_path / "logs"
    code = f"import pathlib,json; p=pathlib.Path({str(root)!r});p.mkdir();(p/'status.json').write_text(json.dumps({{'state':'completed'}}))"
    result = supervisor.supervise(
        [sys.executable, "-c", code],
        root,
        logs,
        poll=0.02,
        refresh=0.02,
        wall=3,
        cleanup=1,
    )
    assert result["returncode"] == 0 and result["stage"] == "completion"
    assert (logs / "runner.log").exists() and (logs / "launch.json").exists()
    with pytest.raises(FileExistsError):
        supervisor.supervise([], root, logs)


@pytest.mark.parametrize("limit", ("deadline", "RAM", "output"))
def test_interrupt_retains_synthetic_failure_and_descendant_accounting(tmp_path, limit):
    root = tmp_path / "run"
    logs = tmp_path / "logs"
    code = f"""import pathlib,signal,time,json,subprocess,sys
p=pathlib.Path({str(root)!r});p.mkdir()
(p/'status.json').write_text(json.dumps({{'state':'running'}}))
def stop(*args):
 (p/'status.json').write_text(json.dumps({{'state':'interrupted'}}))
 raise SystemExit(130)
signal.signal(signal.SIGINT,stop)
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])
(p/'child.json').write_text(json.dumps({{'pid':child.pid}}))
(p/'payload').write_bytes(b'x'*1000)
while True:time.sleep(.02)
"""
    # Let handlers install before checking the deliberately tiny resource limits.
    real_snapshot = supervisor.tree_snapshot

    def ready_snapshot(pid, identity, remembered):
        end = time.monotonic() + 2
        while not (root / "child.json").exists() and time.monotonic() < end:
            time.sleep(0.01)
        return real_snapshot(pid, identity, remembered)

    original = supervisor.tree_snapshot
    supervisor.tree_snapshot = ready_snapshot
    try:
        result = supervisor.supervise(
            [sys.executable, "-c", code],
            root,
            logs,
            poll=0.02,
            refresh=0.02,
            wall=0.7 if limit == "deadline" else 3,
            cleanup=0.3 if limit == "deadline" else 1,
            ram_limit=1 if limit == "RAM" else 4 * supervisor.GIB,
            output_limit=1 if limit == "output" else 5 * supervisor.GIB,
        )
    finally:
        supervisor.tree_snapshot = original
    assert result["stop_reason"] is not None and result["returncode"] == 130
    assert result["stage"] == "failure" and not result["forced_termination"]
    assert json.loads((root / "status.json").read_text())["state"] == "interrupted"
    assert result["peak_observed_ram_bytes"] > 0
    samples = [
        json.loads(line) for line in (logs / "samples.jsonl").read_text().splitlines()
    ]
    assert any(len(s["process_ids"]) >= 2 for s in samples)


def test_bar_labels_inference_only():
    text = supervisor.bar(
        dict(
            stage="bootstrap/scoring",
            completed_batches=13200,
            remaining_inference_seconds=0,
            elapsed_seconds=10,
            batch_rate_per_second=2,
            ram_bytes=100,
            output_bytes=200,
        )
    )
    assert "indeterminate" in text and "100.00% inference" in text


def test_cleanup_deadline_forces_only_stubborn_child(tmp_path):
    root = tmp_path / "run"
    logs = tmp_path / "logs"
    code = (
        "import signal,time;signal.signal(signal.SIGINT,signal.SIG_IGN);time.sleep(10)"
    )
    result = supervisor.supervise(
        [sys.executable, "-c", code],
        root,
        logs,
        poll=0.02,
        refresh=0.02,
        wall=0.5,
        cleanup=0.2,
    )
    assert result["forced_termination"] and not result["cleanup_complete"]
    assert result["returncode"] == -9 and result["stage"] == "failure"
    assert result["elapsed_seconds"] < 2


def test_failed_child_is_reported_without_retry(tmp_path):
    root = tmp_path / "run"
    logs = tmp_path / "logs"
    result = supervisor.supervise(
        [sys.executable, "-c", "raise SystemExit(7)"],
        root,
        logs,
        poll=0.02,
        refresh=0.02,
        wall=3,
        cleanup=1,
    )
    assert result["returncode"] == 7 and result["stage"] == "failure"
    assert not root.exists()
