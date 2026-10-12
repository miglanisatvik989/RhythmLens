"""Synthetic-only evidence: generate all signals in pytest temporary directories.

No fixture contains patient data. Generator seed: 2017; split seed: 42.
"""

import csv
import json
import stat
import zipfile

import numpy as np
import pytest
from scipy.io import savemat

from rhythmlens.audit import audit, decode_record
from rhythmlens.data import (
    LABELS,
    MANIFESTS,
    RAW,
    acquire,
    read_labels,
    safe_extract,
    sha256,
)
from rhythmlens.splits import SPLIT_VERSION, assign_splits, assert_partitions, freeze


def record(directory, number, values=None, gain="1000/mV", adc_zero=0):
    directory.mkdir(parents=True, exist_ok=True)
    rid = f"A{number:05d}"
    if values is None:
        values = np.random.default_rng(2017 + number).integers(
            -1000, 1000, 300, dtype=np.int16
        )
    values = np.asarray(values)
    if np.isfinite(values).all():
        values = values.astype(np.int16)
    finite = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
    checksum = (int(finite.astype(np.int64).sum()) + 32768) % 65536 - 32768
    savemat(directory / f"{rid}.mat", {"val": values.reshape(1, -1)}, format="4")
    (directory / f"{rid}.hea").write_text(
        f"{rid} 1 300 {len(values)}\n"
        f"{rid}.mat 16+24 {gain} 16 {adc_zero} {int(finite[0])} {checksum} 0 ECG\n"
    )
    return directory / f"{rid}.mat", directory / f"{rid}.hea"


def dataset(tmp_path, count=20, modify=None):
    source = tmp_path / "synthetic"
    labels = []
    for class_index, label in enumerate(LABELS):
        for i in range(count):
            number = class_index * count + i + 1
            record(source / "training2017", number)
            labels.append((f"A{number:05d}", label))
    archive_labels = source / "training2017/REFERENCE.csv"
    archive_labels.write_text("".join(f"{rid},{label}\n" for rid, label in labels))
    if modify is not None:
        modify(source / "training2017")
    zip_path = source / "training2017.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in sorted((source / "training2017").iterdir()):
            zf.write(path, path.relative_to(source))
    reference = source / "REFERENCE-v3.csv"
    # Deliberately reverse the CSV: record ID join, never positional join.
    reference.write_text("".join(f"{rid},{label}\n" for rid, label in reversed(labels)))
    root = tmp_path / "project"
    acquire(root, zip_path, reference)
    return root


def rows_for_splits():
    return [
        {
            "record_id": f"{label}{n:03}",
            "subject_id": "unknown",
            "label": label,
            "duplicate_group": f"{label}-{n // 2}",
        }
        for label in LABELS
        for n in range(40)
    ]


def test_shuffled_label_join_and_frozen_roundtrip(tmp_path):
    root = dataset(tmp_path)
    result = audit(root, previews=False)
    assert result["validation_passed"], result["errors"]
    assert result["v3_class_counts"] == dict.fromkeys(LABELS, 20)
    assert result["label_changes"] == []
    with (root / MANIFESTS / "records.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["record_id"] == "A00001" and rows[0]["label"] == "N"
    assert rows[-1]["label"] == "~"
    summary = freeze(root)
    assert summary["split_counts"] == {"train": 56, "validation": 12, "holdout": 12}
    frozen_dir = root / "splits" / SPLIT_VERSION
    hashes = json.loads((frozen_dir / "frozen.json").read_text())["files_sha256"]
    assert all(sha256(frozen_dir / name) == value for name, value in hashes.items())
    with pytest.raises(ValueError, match="refusing overwrite"):
        freeze(root)


@pytest.mark.parametrize(
    "text", ["A00001,X\n", "A00001,N\nA00001,A\n", "A00001\n", "", "../escape,N\n"]
)
def test_invalid_labels(tmp_path, text):
    path = tmp_path / "reference.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_labels(path)


@pytest.mark.parametrize(
    "gain,zero,expected",
    [
        ("1000(100)/mV", 0, [0.0, 1.0, 2.0]),
        ("1000/mV", 100, [0.0, 1.0, 2.0]),
        ("1(100)/uV", 0, [0.0, 1.0, 2.0]),
        ("1000000(100)/V", 0, [0.0, 1.0, 2.0]),
    ],
)
def test_physical_conversion_preserves_originals(tmp_path, gain, zero, expected):
    mat, hea = record(tmp_path, 1, [100, 1100, 2100], gain, zero)
    before = sha256(mat), sha256(hea)
    physical, metadata = decode_record(mat, hea)
    np.testing.assert_allclose(physical, expected)
    assert metadata["physical_units"] == "mV"
    assert before == (sha256(mat), sha256(hea))


def test_content_duplicates_use_decoded_metadata(tmp_path):
    a = record(tmp_path, 1, [1, 2, 3])
    b = record(tmp_path, 2, [1, 2, 3])
    c = record(tmp_path, 3, [1, 2, 3], gain="2000/mV")
    d = record(tmp_path, 4, [2, 4, 6], gain="2000/mV")
    decoded = [decode_record(*pair)[1] for pair in (a, b, c, d)]
    assert decoded[0]["duplicate_group"] == decoded[1]["duplicate_group"]
    assert decoded[0]["duplicate_group"] != decoded[2]["duplicate_group"]
    assert decoded[0]["duplicate_group"] == decoded[3]["duplicate_group"]
    assert (
        decoded[0]["digital_decoding_sha256"] != decoded[3]["digital_decoding_sha256"]
    )


def test_missing_pairs_and_orphans_are_reported(tmp_path):
    root = dataset(tmp_path, 3)
    directory = root / RAW / "extracted/training2017"
    (directory / "A00001.mat").unlink()
    (directory / "A00002.hea").unlink()
    record(directory, 99999)
    result = audit(root, previews=False)
    assert not result["validation_passed"]
    assert result["total_inventory_records"] == 13
    assert result["flag_counts"]["missing_mat_path"] == 1
    assert result["flag_counts"]["missing_header_path"] == 1
    assert result["flag_counts"]["orphan_record_without_v3_label"] == 1
    with pytest.raises(ValueError, match="Audit validation failed"):
        freeze(root)


@pytest.mark.parametrize(
    "values,flag",
    [
        ([0.0, np.nan, 3.0], "nonfinite_signal"),
        ([0.0, np.inf, 3.0], "nonfinite_signal"),
        ([0, -32768, 3], "missing_signal_samples"),
    ],
)
def test_nonfinite_and_missing_signal(tmp_path, values, flag):
    with pytest.raises(ValueError, match=flag):
        decode_record(*record(tmp_path, 1, values))


def test_flat_is_flagged_not_discarded(tmp_path):
    root = dataset(tmp_path, 3)
    directory = root / RAW / "extracted/training2017"
    record(directory, 1, np.zeros(300, dtype=np.int16))
    result = audit(root, previews=False)
    assert result["flag_counts"]["flat"] == 1
    assert (
        result["invalid_records"] == 0
    )  # extraction mismatch blocks freeze, not flatness
    assert result["class_summaries"]["N"]["decoded_records"] == 3
    assert result["class_summaries"]["~"]["decoded_records"] == 3


@pytest.mark.parametrize(
    "old,new,flag",
    [
        ("1 300 300", "1 250 300", "unexpected_sampling_rate"),
        ("1 300 300", "1 300 301", "sample_count_or_channel_shape_mismatch"),
        ("1000/mV", "0/mV", "invalid_gain"),
        ("1000/mV", "1000/Ohm", "unsupported_physical_units"),
        ("1 300 300", "2 300 300", "not_single_channel"),
        ("A00001.mat", "../A00001.mat", "signal_filename_mismatch"),
    ],
)
def test_malformed_header(tmp_path, old, new, flag):
    mat, hea = record(tmp_path, 1)
    hea.write_text(hea.read_text().replace(old, new))
    with pytest.raises(ValueError, match=flag):
        decode_record(mat, hea)


def test_duplicate_leakage_and_determinism():
    rows = rows_for_splits()
    a = assign_splits(rows)
    b = assign_splits(list(reversed(rows)))
    assert a == b
    assert_partitions(a, {r["record_id"] for r in rows})
    assert len(a) == len(rows)
    broken = [dict(r) for r in a]
    broken[0]["split"] = "holdout" if broken[0]["split"] != "holdout" else "train"
    with pytest.raises(ValueError, match="leakage"):
        assert_partitions(broken, {r["record_id"] for r in rows})


def test_conflicting_labels_block():
    rows = rows_for_splits()
    rows[1]["label"] = "A"
    with pytest.raises(ValueError, match="Conflicting labels"):
        assign_splits(rows)


def test_audit_reports_conflicting_duplicate_labels(tmp_path):
    root = dataset(tmp_path, 3)
    directory = root / RAW / "extracted/training2017"
    record(directory, 1, [10, 20, 30])
    record(directory, 4, [10, 20, 30])
    result = audit(root, previews=False)
    assert result["conflicting_duplicate_groups"][0]["record_ids"] == [
        "A00001",
        "A00004",
    ]
    assert "conflicting_labels_within_duplicate_groups" in result["errors"]


@pytest.mark.parametrize(
    "member", ["../escape", "/absolute", "a/../../escape", "C:/escape", "a\\..\\escape"]
)
def test_unsafe_extraction_paths(tmp_path, member):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("safe.txt", "safe")
        zf.writestr(member, "bad")
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, destination)
    assert not destination.exists()  # all members preflighted before any extraction


def test_zip_symlink_and_existing_escape(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        member = zipfile.ZipInfo("link")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(member, "../escape")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path / "output")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("link/escape", "bad")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        safe_extract(archive, destination)


def test_acquisition_reuses_inputs_without_network(tmp_path, monkeypatch):
    root = dataset(tmp_path, 3)

    def forbidden(*args, **kwargs):
        pytest.fail("unexpected network request")

    monkeypatch.setattr("rhythmlens.data.urlopen", forbidden)
    before = sha256(root / MANIFESTS / "provenance.json")
    provenance = acquire(root)
    assert all(
        entry["downloaded_at_utc"] is None for entry in provenance["files"].values()
    )
    assert sha256(root / MANIFESTS / "provenance.json") == before


def test_changed_inputs_block_freeze(tmp_path):
    root = dataset(tmp_path, 3)
    assert audit(root, previews=False)["validation_passed"]
    with (root / RAW / "extracted/training2017/A00001.hea").open("a") as stream:
        stream.write("# changed after audit\n")
    with pytest.raises(ValueError, match="Audited input changed"):
        freeze(root)


def test_class_coverage_requires_independent_groups():
    rows = rows_for_splits()[:2]
    with pytest.raises(ValueError, match="class coverage"):
        assign_splits(rows)


def test_valid_duplicates_and_flat_records_survive_freeze(tmp_path):
    def modify(directory):
        record(directory, 1, [1, 2, 3])
        record(directory, 2, [1, 2, 3])
        record(directory, 3, [0, 0, 0])

    root = dataset(tmp_path, count=20, modify=modify)
    result = audit(root, previews=False)
    assert result["validation_passed"], result["errors"]
    assert result["flag_counts"]["flat"] == 1
    assert len(result["duplicate_groups"]) == 1
    summary = freeze(root)
    assert summary["records"] == 80
    partitions = []
    for path in (root / "splits" / SPLIT_VERSION).glob("*.csv"):
        with path.open() as stream:
            partitions.extend(csv.DictReader(stream))
    assignment = {r["record_id"]: r["split"] for r in partitions}
    assert assignment["A00001"] == assignment["A00002"]
    assert "A00003" in assignment
    with pytest.raises(ValueError, match="audit snapshot"):
        audit(root, previews=False)


def test_conflicts_in_unchanged_archive_block_freeze(tmp_path):
    def modify(directory):
        record(directory, 1, [1, 2, 3])
        record(directory, 21, [1, 2, 3])

    root = dataset(tmp_path, modify=modify)
    result = audit(root, previews=False)
    assert result["errors"] == ["conflicting_labels_within_duplicate_groups"]
    with pytest.raises(ValueError, match="conflicting_labels"):
        freeze(root)
    assert not (root / "splits" / SPLIT_VERSION).exists()


def test_invalid_mat_and_offset(tmp_path):
    mat, hea = record(tmp_path, 1)
    hea.write_text(hea.read_text().replace("16+24", "16+26"))
    with pytest.raises((ValueError, IndexError)):
        decode_record(mat, hea)
    mat.write_bytes(b"not a MAT file")
    with pytest.raises(Exception):
        decode_record(mat, hea)


def test_duplicate_zip_targets_rejected(tmp_path):
    path = tmp_path / "repeated.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("same", "one")
        with pytest.warns(UserWarning, match="Duplicate"):
            zf.writestr("same", "two")
    with pytest.raises(ValueError, match="Duplicate"):
        safe_extract(path, tmp_path / "extracted")


def test_new_orphan_after_audit_blocks_freeze(tmp_path):
    root = dataset(tmp_path, count=3)
    assert audit(root, previews=False)["validation_passed"]
    record(root / RAW / "extracted/training2017", 99999)
    with pytest.raises(ValueError, match="inventory changed"):
        freeze(root)


def test_zero_checksum_and_format16_initial_value_are_explicit_advisories(tmp_path):
    mat, hea = record(tmp_path, 1, [10, 20, 30])
    hea.write_text(hea.read_text().replace("0 10 60 0 ECG", "0 999 0 0 ECG"))
    physical, metadata = decode_record(mat, hea)
    assert metadata["advisories"] == [
        "initial_value_mismatch_format16",
        "zero_header_checksum_unverified",
    ]
    assert metadata["actual_initial_value"] == 10
    assert metadata["computed_checksum_unsigned"] == 60
    np.testing.assert_allclose(physical, [0.01, 0.02, 0.03])
    hea.write_text(hea.read_text().replace("999 0 0 ECG", "999 61 0 ECG"))
    with pytest.raises(ValueError, match="signal_checksum_mismatch"):
        decode_record(mat, hea)
