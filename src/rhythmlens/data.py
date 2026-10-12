"""Acquisition and integrity helpers for the public Challenge 2017 training data."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from urllib.request import urlopen
import zipfile

LABELS = ("N", "A", "O", "~")
BASE_URL = "https://physionet.org/files/challenge-2017/1.0.0/"
S3_URL = "https://physionet-open.s3.amazonaws.com/challenge-2017/1.0.0/"
DATASET_URL = "https://physionet.org/content/challenge-2017/1.0.0/"
LICENSE = {
    "name": "Open Data Commons Attribution License v1.0",
    "spdx": "ODC-By-1.0",
    "url": "https://physionet.org/content/challenge-2017/view-license/1.0.0/",
}
ATTRIBUTION = {
    "dataset": "PhysioNet/Computing in Cardiology Challenge 2017; recordings donated by AliveCor",
    "dataset_doi": "10.13026/d3hm-sf11",
    "publication": "Clifford GD et al. AF classification from a short single lead ECG recording. Computing in Cardiology (2017).",
    "publication_doi": "10.22489/CinC.2017.065-469",
    "physionet_citation": "Pollard T et al. PhysioNet as a global platform for biomedical research. Nature Health (2026).",
    "physionet_doi": "10.1038/s44360-026-00096-z",
}
RAW = Path("data/raw/challenge2017")
MANIFESTS = Path("data/manifests/challenge2017-v3")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically; a failed write cannot leave a partially written manifest.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def read_labels(path: Path) -> dict[str, str]:
    """Read a headerless reference table, rejecting ambiguous or invalid keys."""
    labels = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for number, row in enumerate(csv.reader(stream), 1):
            if len(row) != 2:
                raise ValueError(f"{path.name}:{number}: expected record ID,label")
            record_id, label = (item.strip() for item in row)
            if not re.fullmatch(r"A[0-9]{5}", record_id) or label not in LABELS:
                raise ValueError(f"{path.name}:{number}: invalid ID or label: {row!r}")
            if record_id in labels:
                raise ValueError(
                    f"{path.name}:{number}: duplicate label ID {record_id}"
                )
            labels[record_id] = label
    if not labels:
        raise ValueError(f"{path.name}: empty reference")
    return labels


def safe_extract(archive: Path, destination: Path) -> None:
    """Preflight every ZIP member; reject traversal, links and duplicate targets."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        seen = set()
        for info in zf.infolist():
            name = info.filename
            path = PurePosixPath(name)
            mode = info.external_attr >> 16
            if (
                not name
                or "\\" in name
                or ":" in name
                or path.is_absolute()
                or ".." in path.parts
                or stat.S_ISLNK(mode)
                or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))
            ):
                raise ValueError(f"Unsafe ZIP member: {name!r}")
            target = destination.joinpath(*path.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"ZIP member escapes destination: {name!r}")
            if target in seen or target.is_symlink():
                raise ValueError(f"Duplicate or linked ZIP target: {name!r}")
            seen.add(target)
        # Limits exceed the official archive but prevent accidental decompression bombs.
        if len(seen) > 50000 or sum(i.file_size for i in zf.infolist()) > 2_000_000_000:
            raise ValueError("Archive exceeds extraction limits")
        destination.mkdir(parents=True, exist_ok=True)
        for info in zf.infolist():
            target = destination / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def acquire(
    root: Path,
    local_archive: Path | None = None,
    local_reference: Path | None = None,
    mirror: str = "physionet",
) -> dict:
    """Cache only the two official inputs and preserve truthful acquisition times."""
    if mirror not in ("physionet", "s3"):
        raise ValueError("Unknown official mirror")
    download_base = S3_URL if mirror == "s3" else BASE_URL
    root = root.resolve()
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    provenance_path = root / MANIFESTS / "provenance.json"
    provenance = (
        json.loads(provenance_path.read_text())
        if provenance_path.exists()
        else {
            "schema_version": 1,
            "dataset_version": "1.0.0",
            "label_version": "REFERENCE-v3",
            "dataset_url": DATASET_URL,
            "license": LICENSE,
            "attribution": ATTRIBUTION,
            "files": {},
        }
    )
    for name, local in (
        ("training2017.zip", local_archive),
        ("REFERENCE-v3.csv", local_reference),
    ):
        target = raw / name
        prior = provenance["files"].get(name)
        downloaded_at = None
        method = "existing-local"
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".part")
            try:
                if local is not None:
                    shutil.copyfile(local, partial)
                    method = "local-copy"
                else:
                    with (
                        urlopen(download_base + name, timeout=120) as response,
                        partial.open("wb") as dst,
                    ):
                        shutil.copyfileobj(response, dst)
                    downloaded_at = utc_now()
                    method = "download"
                if name.endswith(".zip"):
                    with zipfile.ZipFile(partial) as archive:
                        if archive.testzip() is not None:
                            raise ValueError("Archive CRC verification failed")
                else:
                    read_labels(partial)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
        digest = sha256(target)
        if prior and digest != prior["sha256"]:
            raise ValueError(
                f"Cached {name} differs from recorded provenance; investigate before proceeding"
            )
        if not prior:
            provenance["files"][name] = {
                "path": relative(target, root),
                "source_url": download_base + name
                if method == "download"
                else BASE_URL + name,
                "official_source_url": BASE_URL + name,
                "sha256": digest,
                "bytes": target.stat().st_size,
                "downloaded_at_utc": downloaded_at,
                "observed_at_utc": utc_now(),
                "acquisition_method": method,
                "download_time_note": "Unknown for pre-existing/local copies"
                if downloaded_at is None
                else None,
            }
        write_json(provenance_path, provenance)
    read_labels(raw / "REFERENCE-v3.csv")
    extracted = raw / "extracted"
    marker = extracted / ".archive-sha256"
    digest = provenance["files"]["training2017.zip"]["sha256"]
    if extracted.exists():
        if not marker.exists() or marker.read_text().strip() != digest:
            raise ValueError(
                "Existing extraction has no matching archive marker; refusing to overwrite"
            )
    else:
        with tempfile.TemporaryDirectory(prefix="extract-", dir=raw) as temporary:
            staged = Path(temporary) / "extracted"
            safe_extract(raw / "training2017.zip", staged)
            (staged / ".archive-sha256").write_text(digest + "\n")
            os.rename(staged, extracted)
    return provenance
