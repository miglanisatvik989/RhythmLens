"""Explicit NSTDB 1.0.0 noise-only acquisition and audit; no Challenge loader."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

import numpy as np

BASE_URL = "https://physionet.org/files/nstdb/1.0.0/"
RECORDS = ("bw", "ma", "em")
FILES = ("RECORDS", "SHA256SUMS.txt", "nstdb.txt") + tuple(
    f"{record}.{ext}" for record in RECORDS for ext in ("dat", "hea")
)
RAW = Path("data/raw/nstdb/1.0.0")
MANIFEST = Path("data/manifests/nstdb-v1")
SAMPLES = 650000
SEMANTICS = "digital_minus_adc_zero_unitless_noise_shape_not_calibrated_voltage"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_new_json(path: Path, value: dict) -> None:
    """Exclusive creation; preserve partial writes rather than overwrite/retry."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
        stream.write("\n")


def validate_files(blobs: dict[str, bytes]) -> dict[str, str]:
    if set(blobs) != set(FILES):
        raise ValueError("Expected exactly the nine NSTDB noise-only files")
    members = blobs["RECORDS"].decode("ascii").splitlines()
    if len(members) != len(set(members)) or not set(RECORDS) <= set(members):
        raise ValueError("RECORDS missing required membership or has duplicates")
    checksums = {}
    for line in blobs["SHA256SUMS.txt"].decode("ascii").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](\S+)", line)
        if not match:
            raise ValueError("Malformed publisher checksum index")
        name = match[2].removeprefix("./")
        if name in checksums or "/" in name or name in (".", ".."):
            raise ValueError("Ambiguous or unsafe checksum index member")
        checksums[name] = match[1].lower()
    hashes = {name: digest(value) for name, value in blobs.items()}
    for name in FILES:
        if name != "SHA256SUMS.txt" and checksums.get(name) != hashes[name]:
            raise ValueError(f"Publisher checksum missing/mismatch: {name}")
    return hashes


def read_files(raw: Path) -> dict[str, bytes]:
    blobs = {}
    for name in FILES:
        path = raw / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing/nonregular required file: {name}")
        blobs[name] = path.read_bytes()
    return blobs


def acquire(raw: Path, manifest: Path, *, local: Path | None, download: bool) -> dict:
    """Only an explicit local source OR explicit download can acquire bytes."""
    if (local is not None) == download or type(download) is not bool:
        raise ValueError("Choose exactly one explicit local import or download")
    if raw.exists() or raw.is_symlink() or manifest.exists() or manifest.is_symlink():
        raise FileExistsError("Destination exists; preserve it, never overwrite/retry")
    if raw.resolve() == manifest.resolve() or (
        raw.resolve() in manifest.resolve().parents
        or manifest.resolve() in raw.resolve().parents
    ):
        raise ValueError("Raw and provenance directories must be separate")
    blobs, timestamps = {}, {}
    for name in FILES:
        if local is not None:
            source = local / name
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Missing/nonregular local import member: {name}")
            blobs[name] = source.read_bytes()
        else:
            url = BASE_URL + name
            with urlopen(url, timeout=60) as response:
                if response.geturl() != url:
                    raise ValueError("Unexpected redirect from versioned source URL")
                blobs[name] = response.read(10_000_001)
                if len(blobs[name]) > 10_000_000:
                    raise ValueError("Unexpectedly oversized release file")
        timestamps[name] = now()
    hashes = validate_files(blobs)
    provenance = {
        "schema": "rhythmlens-nstdb-acquisition-v1",
        "dataset": "nstdb",
        "version": "1.0.0",
        "doi": "10.13026/C2HS3T",
        "license": "ODC-By-1.0",
        "license_url": "https://physionet.org/content/nstdb/view-license/1.0.0/",
        "attribution": "Moody GB, Muldrow W, Mark RG. A noise stress test for arrhythmia detectors. Computers in Cardiology 11:381-384 (1984); MIT-BIH Noise Stress Test Database, PhysioNet.",
        "method": "explicit_local_import"
        if local is not None
        else "explicit_https_download",
        "local_source": str(local.resolve()) if local is not None else None,
        "checksum_trust": "owner_supplied_publisher_index"
        if local is not None
        else "versioned_PhysioNet_HTTPS",
        "checksum_index_self_authenticated": False,
        "files": {
            name: {
                "source_url": BASE_URL + name,
                "acquired_utc": timestamps[name],
                "bytes": len(blobs[name]),
                "sha256": hashes[name],
                "publisher_checksum_verified": name != "SHA256SUMS.txt",
            }
            for name in FILES
        },
        "audit": "pending",
        "external_archive": "pending_destination_unspecified",
    }
    # Validate all bytes first, then exclusively reserve both destinations.
    raw.mkdir(parents=True, exist_ok=False)
    manifest.mkdir(parents=True, exist_ok=False)
    for name, value in blobs.items():
        with (raw / name).open("xb") as stream:
            stream.write(value)
    write_new_json(manifest / "provenance.json", provenance)
    return provenance


def decode_212(value: bytes, samples: int) -> np.ndarray:
    """Two channels, signed 12-bit packed triples. Keep -2048 visible as missing."""
    if type(samples) is not int or samples <= 0 or len(value) != samples * 3:
        raise ValueError("Format-212 byte length mismatch")
    packed = np.frombuffer(value, dtype=np.uint8).reshape(-1, 3).astype(np.int16)
    digital = np.column_stack(
        (
            packed[:, 0] | ((packed[:, 1] & 15) << 8),
            packed[:, 2] | ((packed[:, 1] >> 4) << 8),
        )
    )
    return np.where(digital >= 2048, digital - 4096, digital).astype(np.int16)


def decode_record(
    record: str, header: bytes, payload: bytes
) -> tuple[np.ndarray, dict]:
    if record not in RECORDS:
        raise ValueError("Only bw/ma/em noise records supported")
    lines = [
        line.split()
        for line in header.decode("ascii").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(lines) != 3 or lines[0] != [record, "2", "360", str(SAMPLES)]:
        raise ValueError(f"Unexpected {record} record/channel/rate/length header")
    fields = []
    for channel, line in enumerate(lines[1:]):
        if (
            len(line) != 9
            or line[:5] != [f"{record}.dat", "212", "0", "12", "0"]
            or line[7:] != ["0", f"noise{channel + 1}"]
        ):
            raise ValueError(f"Unsupported {record} channel fields")
        initial, checksum = int(line[5]), int(line[6])
        if not -2048 <= initial <= 2047 or not -32768 <= checksum <= 32767:
            raise ValueError("Header initial/checksum outside signed range")
        fields.append(
            {
                "name": line[8],
                "format": 212,
                "gain": 0,
                "adc_resolution": 12,
                "adc_zero": 0,
                "initial": initial,
                "checksum": checksum,
                "block_size": 0,
            }
        )
    digital = decode_212(payload, SAMPLES)
    missing = (digital == -2048).sum(axis=0).tolist()
    checksums = ((digital.sum(axis=0, dtype=np.int64) + 32768) % 65536 - 32768).tolist()
    problems = []
    if any(missing):
        problems.append("missing_value_sentinel")
    if checksums != [f["checksum"] for f in fields]:
        problems.append("header_checksum_mismatch")
    if digital[0].tolist() != [f["initial"] for f in fields]:
        problems.append("header_initial_mismatch")
    info = {
        "record": record,
        "channels": fields,
        "samples_per_channel": SAMPLES,
        "sampling_rate_hz": 360,
        "missing_samples_per_channel": missing,
        "computed_checksums": checksums,
        "semantics": SEMANTICS,
        "errors": problems,
        "status": "failed" if problems else "passed",
    }
    return digital, info


def verified_blobs(raw: Path, provenance: dict) -> tuple[dict, dict]:
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema") != "rhythmlens-nstdb-acquisition-v1"
        or provenance.get("dataset") != "nstdb"
        or provenance.get("version") != "1.0.0"
        or not isinstance(provenance.get("files"), dict)
        or set(provenance.get("files", {})) != set(FILES)
    ):
        raise ValueError("Unsupported/incomplete acquisition provenance")
    blobs = read_files(raw)
    hashes = validate_files(blobs)
    for name in FILES:
        entry = provenance["files"][name]
        if (
            not isinstance(entry, dict)
            or entry.get("source_url") != BASE_URL + name
            or entry.get("sha256") != hashes[name]
            or type(entry.get("bytes")) is not int
            or entry.get("bytes") != len(blobs[name])
            or entry.get("publisher_checksum_verified")
            is not (name != "SHA256SUMS.txt")
        ):
            raise ValueError(f"Acquisition provenance mismatch: {name}")
    return blobs, hashes


def audit(raw: Path, provenance: dict) -> dict:
    """Read only NSTDB; collect failures, never substitute or discard a record."""
    from .noise import prepare_pool

    report = {
        "schema": "rhythmlens-nstdb-audit-v1",
        "created_utc": now(),
        "status": "failed",
        "records": [],
        "roles": [],
        "errors": [],
        "subject_ids": "unavailable",
        "independence": "temporal_not_subject_or_source",
        "numpy_version": np.__version__,
    }
    try:
        blobs, hashes = verified_blobs(raw, provenance)
    except (ValueError, OSError, UnicodeError) as error:
        report["errors"].append(str(error))
        return report
    report["file_sha256"] = hashes
    report["provenance_sha256"] = digest(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    )
    for record in RECORDS:
        try:
            digital, info = decode_record(
                record, blobs[f"{record}.hea"], blobs[f"{record}.dat"]
            )
            report["records"].append(info)
            if info["errors"]:
                report["errors"].append(f"{record}: {info['errors']}")
                continue
            for role in ("augmentation-train", "development-reserve", "stress"):
                pool = prepare_pool(
                    digital,
                    record=record,
                    role=role,
                    source_sha256=hashes[f"{record}.dat"],
                )
                report["roles"].append(pool.metadata())
        except (ValueError, OSError, UnicodeError) as error:
            report["errors"].append(f"{record}: {error}")
            existing = next(
                (r for r in report["records"] if r["record"] == record), None
            )
            if existing is None:
                report["records"].append(
                    {"record": record, "status": "failed", "errors": [str(error)]}
                )
            else:
                existing["status"] = "failed"
                existing["errors"].append(str(error))
    if not report["errors"]:
        report["status"] = "passed"
    return report


def load_pools(raw: Path, provenance: dict, role: str) -> dict:
    """Revalidate every source record before preparing any usable adapter pools."""
    from .noise import prepare_pool

    blobs, hashes = verified_blobs(raw, provenance)
    digital_records = {}
    for record in RECORDS:
        digital, info = decode_record(
            record, blobs[f"{record}.hea"], blobs[f"{record}.dat"]
        )
        if info["errors"]:
            raise ValueError(f"NSTDB technical failure: {record}: {info['errors']}")
        digital_records[record] = digital
    return {
        record: prepare_pool(
            value, record=record, role=role, source_sha256=hashes[f"{record}.dat"]
        )
        for record, value in digital_records.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    acquire_parser = commands.add_parser("acquire")
    source = acquire_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--local", type=Path)
    source.add_argument("--download", action="store_true")
    acquire_parser.add_argument("--raw", type=Path, default=RAW)
    acquire_parser.add_argument("--manifest", type=Path, default=MANIFEST)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--raw", type=Path, default=RAW)
    audit_parser.add_argument(
        "--provenance", type=Path, default=MANIFEST / "provenance.json"
    )
    audit_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "acquire":
            acquire(args.raw, args.manifest, local=args.local, download=args.download)
        else:
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError("Audit output exists; never overwrite/retry")
            provenance = json.loads(args.provenance.read_text())
            report = audit(args.raw, provenance)
            args.output.mkdir(parents=True, exist_ok=False)
            write_new_json(args.output / "audit.json", report)
            if report["status"] == "passed":
                write_new_json(
                    args.output / "roles.json",
                    {
                        "schema": "rhythmlens-nstdb-roles-v1",
                        "audit_sha256": digest(
                            (args.output / "audit.json").read_bytes()
                        ),
                        "roles": report["roles"],
                    },
                )
            write_new_json(
                args.output / "sha256.json",
                {
                    name: digest((args.output / name).read_bytes())
                    for name in ("audit.json", "roles.json")
                    if (args.output / name).exists()
                },
            )
            if report["status"] != "passed":
                print(json.dumps({"status": "failed", "errors": report["errors"]}))
                return 1
    except (ValueError, OSError) as error:
        parser.exit(1, f"NSTDB blocked: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
