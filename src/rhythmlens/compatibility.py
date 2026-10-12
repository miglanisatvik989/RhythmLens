"""Reviewed decoder semantics, independent of historical audit environment hashes."""

import json
from pathlib import Path
from rhythmlens.data import sha256


def validate_decoder_contract():
    path = Path(__file__).with_name("data_compatibility.json")
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != 1
        or value.get("contract_version") != "challenge2017-physical-v1"
        or value.get("sampling_rate_hz") != 300
        or value.get("units") != "mV"
    ):
        raise ValueError("Unsupported decoder compatibility contract")
    for name in ("audit.py", "data.py"):
        if sha256(path.parent / name) != value["decoder_sha256"].get(name):
            raise ValueError(f"Unsupported decoder implementation: {name}")
    return {"contract": value, "contract_sha256": sha256(path)}
