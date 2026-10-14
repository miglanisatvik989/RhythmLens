"""Generated format-212 noise only: no network, clinical ECG, weights or outputs."""

import hashlib
import json

import numpy as np
import pytest

from rhythmlens import nstdb
from rhythmlens.noise import prepare_pool


def pack(value):
    value = np.asarray(value, dtype=np.int16) & 4095
    packed = np.column_stack(
        (
            value[:, 0] & 255,
            (value[:, 0] >> 8) | ((value[:, 1] >> 8) << 4),
            value[:, 1] & 255,
        )
    )
    return packed.astype(np.uint8).tobytes()


def header(record, value):
    checksums = (value.sum(axis=0, dtype=np.int64) + 32768) % 65536 - 32768
    lines = [f"{record} 2 360 {nstdb.SAMPLES}"]
    lines += [
        f"{record}.dat 212 0 12 0 {value[0, ch]} {checksums[ch]} 0 noise{ch + 1}"
        for ch in (0, 1)
    ]
    return ("\n".join(lines) + "\n").encode()


def index(blobs):
    blobs["SHA256SUMS.txt"] = "".join(
        f"{hashlib.sha256(value).hexdigest()}  {name}\n"
        for name, value in blobs.items()
        if name != "SHA256SUMS.txt"
    ).encode()
    return blobs


@pytest.fixture(scope="module")
def synthetic_blobs():
    t = np.arange(nstdb.SAMPLES, dtype=np.int32)
    value = np.column_stack((t % 201 - 100, (t * 7) % 301 - 150)).astype(np.int16)
    blobs = {
        "RECORDS": b"118e00\nbw\nma\nem\n",
        "nstdb.txt": b"SYNTHETIC fixture, not real NSTDB\n",
    }
    for record in nstdb.RECORDS:
        blobs[f"{record}.dat"] = pack(value)
        blobs[f"{record}.hea"] = header(record, value)
    return index(blobs)


@pytest.fixture
def local_source(tmp_path, synthetic_blobs):
    source = tmp_path / "source"
    source.mkdir()
    for name, value in synthetic_blobs.items():
        (source / name).write_bytes(value)
    return source


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Network is forbidden in synthetic tests")

    monkeypatch.setattr(nstdb, "urlopen", blocked)


def test_known_212_bytes_and_sentinel():
    # Hand-computed triples: extremes, mixed signs, and the WFDB missing marker.
    payload = bytes([0, 0x78, 255, 255, 0x0F, 0, 1, 0x80, 0])
    np.testing.assert_array_equal(
        nstdb.decode_212(payload, 3), [[-2048, 2047], [-1, 0], [1, -2048]]
    )
    with pytest.raises(ValueError, match="length"):
        nstdb.decode_212(payload[:-1], 3)


def test_decoder_matches_independent_wfdb_digital(local_source, synthetic_blobs):
    import wfdb

    expected = wfdb.rdrecord(str(local_source / "bw"), physical=False).d_signal
    actual, info = nstdb.decode_record(
        "bw", synthetic_blobs["bw.hea"], synthetic_blobs["bw.dat"]
    )
    np.testing.assert_array_equal(actual, expected)
    assert info["status"] == "passed"
    assert info["channels"][0]["gain"] == 0
    assert "not_calibrated_voltage" in info["semantics"]
    assert info["samples_per_channel"] == 650000


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("212", "16"),
        ("360", "300"),
        ("650000", "649999"),
        ("noise2", "ECG"),
        (" 0 12 ", " 200 12 "),
    ],
)
def test_header_contract(synthetic_blobs, field, replacement):
    with pytest.raises(ValueError):
        nstdb.decode_record(
            "bw",
            synthetic_blobs["bw.hea"].replace(field.encode(), replacement.encode()),
            synthetic_blobs["bw.dat"],
        )


def test_missing_initial_and_checksum_are_technical_failures(synthetic_blobs):
    value = nstdb.decode_212(synthetic_blobs["bw.dat"], nstdb.SAMPLES)
    value[3, 1] = -2048
    _, info = nstdb.decode_record("bw", header("bw", value), pack(value))
    assert info["missing_samples_per_channel"] == [0, 1]
    assert info["errors"] == ["missing_value_sentinel"]
    with pytest.raises(ValueError, match="sentinel"):
        prepare_pool(value, record="bw", role="stress", source_sha256="a" * 64)
    value[0, 0] += 1
    _, info = nstdb.decode_record("bw", synthetic_blobs["bw.hea"], pack(value))
    assert "header_initial_mismatch" in info["errors"]
    assert "header_checksum_mismatch" in info["errors"]


def test_acquisition_original_bytes_provenance_and_nonoverwrite(
    tmp_path, local_source, synthetic_blobs
):
    raw, manifest = tmp_path / "raw", tmp_path / "manifest"
    result = nstdb.acquire(raw, manifest, local=local_source, download=False)
    assert result == json.loads((manifest / "provenance.json").read_text())
    assert result["license"] == "ODC-By-1.0"
    assert result["checksum_index_self_authenticated"] is False
    assert result["audit"] == "pending"
    for name in nstdb.FILES:
        assert (raw / name).read_bytes() == synthetic_blobs[name]
        entry = result["files"][name]
        assert entry["source_url"] == nstdb.BASE_URL + name
        assert entry["sha256"] == hashlib.sha256(synthetic_blobs[name]).hexdigest()
        assert entry["bytes"] == len(synthetic_blobs[name])
        assert entry["acquired_utc"].endswith("+00:00")
    with pytest.raises(FileExistsError):
        nstdb.acquire(raw, manifest, local=local_source, download=False)
    assert nstdb.read_files(raw) == synthetic_blobs


@pytest.mark.parametrize(
    "failure", ["checksum", "membership", "missing", "duplicate", "unsafe"]
)
def test_acquisition_failures_publish_nothing(
    tmp_path, local_source, synthetic_blobs, failure
):
    blobs = dict(synthetic_blobs)
    if failure == "checksum":
        blobs["bw.dat"] += b"bad"
    elif failure == "membership":
        blobs["RECORDS"] = b"bw\nma\n"
        index(blobs)
    elif failure == "missing":
        (local_source / "ma.hea").unlink()
    elif failure == "duplicate":
        blobs["SHA256SUMS.txt"] += blobs["SHA256SUMS.txt"].splitlines(keepends=True)[0]
    else:
        blobs["SHA256SUMS.txt"] += b"0" * 64 + b"  ../escape\n"
    for name, value in blobs.items():
        if failure != "missing" or name != "ma.hea":
            (local_source / name).write_bytes(value)
    raw, manifest = tmp_path / "raw", tmp_path / "manifest"
    with pytest.raises(ValueError):
        nstdb.acquire(raw, manifest, local=local_source, download=False)
    assert not raw.exists() and not manifest.exists()


def test_explicit_source_required_and_collision_precedes_read(tmp_path):
    raw, manifest = tmp_path / "raw", tmp_path / "manifest"
    for local, download in ((None, False), (tmp_path, True)):
        with pytest.raises(ValueError):
            nstdb.acquire(raw, manifest, local=local, download=download)
    raw.mkdir()
    (raw / "partial").write_text("preserve")
    with pytest.raises(FileExistsError):
        nstdb.acquire(raw, manifest, local=tmp_path / "nonexistent", download=False)
    assert (raw / "partial").read_text() == "preserve"


def test_download_is_explicit_exact_urls_synthetic_transport(
    tmp_path, synthetic_blobs, monkeypatch
):
    urls = []

    class Response:
        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def geturl(self):
            return self.url

        def read(self, size):
            return synthetic_blobs[self.url.rsplit("/", 1)[1]]

    def fake_open(url, timeout):
        assert timeout == 60
        urls.append(url)
        return Response(url)

    monkeypatch.setattr(nstdb, "urlopen", fake_open)
    result = nstdb.acquire(
        tmp_path / "raw", tmp_path / "manifest", local=None, download=True
    )
    assert urls == [nstdb.BASE_URL + name for name in nstdb.FILES]
    assert result["method"] == "explicit_https_download"


def test_audit_cli_manifest_integrity_and_collision(tmp_path, local_source):
    raw, manifest = tmp_path / "raw", tmp_path / "manifest"
    assert (
        nstdb.main(
            [
                "acquire",
                "--local",
                str(local_source),
                "--raw",
                str(raw),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    output = tmp_path / "audit"
    argv = [
        "audit",
        "--raw",
        str(raw),
        "--provenance",
        str(manifest / "provenance.json"),
        "--output",
        str(output),
    ]
    assert nstdb.main(argv) == 0
    report = json.loads((output / "audit.json").read_text())
    assert report["status"] == "passed"
    assert len(report["records"]) == 3 and len(report["roles"]) == 9
    assert {r["samples_per_channel"] for r in report["roles"]} == {179400, 143400}
    for name, expected in json.loads((output / "sha256.json").read_text()).items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    before = (output / "audit.json").read_bytes()
    with pytest.raises(SystemExit):
        nstdb.main(argv)
    assert (output / "audit.json").read_bytes() == before
    (raw / "em.dat").write_bytes(b"corrupted")
    failed_output = tmp_path / "failed-audit"
    argv[-1] = str(failed_output)
    assert nstdb.main(argv) == 1
    assert json.loads((failed_output / "audit.json").read_text())["status"] == "failed"
    assert not (failed_output / "roles.json").exists()


def test_audit_missing_sentinel_is_retained_and_blocks_pools(
    tmp_path, local_source, synthetic_blobs
):
    blobs = dict(synthetic_blobs)
    value = nstdb.decode_212(blobs["em.dat"], nstdb.SAMPLES)
    value[7, 0] = -2048
    blobs["em.dat"], blobs["em.hea"] = pack(value), header("em", value)
    index(blobs)
    for name, blob in blobs.items():
        (local_source / name).write_bytes(blob)
    raw, manifest = tmp_path / "raw", tmp_path / "manifest"
    provenance = nstdb.acquire(raw, manifest, local=local_source, download=False)
    report = nstdb.audit(raw, provenance)
    assert report["status"] == "failed"
    assert len(report["records"]) == 3
    assert report["records"][2]["missing_samples_per_channel"] == [1, 0]
    with pytest.raises(ValueError, match="technical failure"):
        nstdb.load_pools(raw, provenance, "stress")


def test_verified_loader_and_provenance_tamper(tmp_path, local_source):
    raw, manifest = tmp_path / "raw", tmp_path / "manifest"
    provenance = nstdb.acquire(raw, manifest, local=local_source, download=False)
    pools = nstdb.load_pools(raw, provenance, "stress")
    assert set(pools) == {"bw", "ma", "em"}
    provenance["files"]["bw.dat"]["sha256"] = "0" * 64
    assert nstdb.audit(raw, provenance)["status"] == "failed"
    with pytest.raises(ValueError, match="provenance mismatch"):
        nstdb.load_pools(raw, provenance, "stress")


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {},
        {
            "schema": "rhythmlens-nstdb-acquisition-v1",
            "dataset": "nstdb",
            "version": "1.0.0",
            "files": None,
        },
    ],
)
def test_malformed_provenance_fails_without_input_access(tmp_path, bad, monkeypatch):
    def blocked(*args):
        pytest.fail("Invalid provenance must be rejected before input access")

    monkeypatch.setattr(nstdb, "read_files", blocked)
    result = nstdb.audit(tmp_path / "missing", bad)
    assert result["status"] == "failed" and result["errors"]


def test_structural_audit_failure_keeps_three_record_inventory(
    tmp_path, local_source, synthetic_blobs
):
    blobs = dict(synthetic_blobs)
    blobs["ma.hea"] = blobs["ma.hea"].replace(b"360", b"300")
    index(blobs)
    for name, blob in blobs.items():
        (local_source / name).write_bytes(blob)
    raw, manifest = tmp_path / "raw", tmp_path / "manifest"
    provenance = nstdb.acquire(raw, manifest, local=local_source, download=False)
    result = nstdb.audit(raw, provenance)
    assert result["status"] == "failed"
    assert [r["record"] for r in result["records"]] == list(nstdb.RECORDS)
    assert result["records"][1]["status"] == "failed"
    assert all(role["record"] != "ma" for role in result["roles"])
