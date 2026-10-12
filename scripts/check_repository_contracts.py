"""Raw-data-free ignore, lock-source and portability checks."""

from pathlib import Path
import subprocess
import tomllib
from urllib.parse import urlparse

root = Path(__file__).resolve().parents[1]
ignored = [
    ".venv/pyvenv.cfg",
    "environments/cuda128/.venv/pyvenv.cfg",
    "data/raw/example.mat",
    "weights/example.pt",
    "outputs/example/history.json",
]
trackable = [
    "configs/reference.yaml",
    "configs/temporal.yaml",
    "data/manifests/challenge2017-v3/records.csv",
    "splits/challenge2017-v3-seed42-v1/train.csv",
    "reports/pretraining-readiness.md",
]
for name in ignored + trackable:
    r = subprocess.run(["git", "check-ignore", "--no-index", "-q", name], cwd=root)
    assert r.returncode == (0 if name in ignored else 1), name
print(
    "Ignore rules: environments, ECG, weights, outputs ignored; configs/manifests/splits/reports trackable"
)
for file in ("uv.lock", "environments/cuda128/uv.lock"):
    text = (root / file).read_text()
    assert "/home/" not in text and "/tmp/" not in text
    lock = tomllib.loads(text)

    def visit(value):
        if isinstance(value, dict):
            for v in value.values():
                visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)
        elif isinstance(value, str) and "://" in value:
            url = urlparse(value)
            assert not url.username and not url.password and not url.query, value

    visit(lock)
    versions = [p["version"] for p in lock["package"] if p["name"] == "torch"]
    assert (
        ("2.8.0+cpu" in versions)
        if file == "uv.lock"
        else (versions == ["2.8.0+cu128"])
    )
    print(file, "portable credential-free lock; Torch:", versions)
