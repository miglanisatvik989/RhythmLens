> Historical workflow snapshot. Use ../reproduction.md for current instructions.

# RhythmLens CUDA 12.8 environment

Separate Linux x86_64 Python 3.11 environment for an NVIDIA GPU. The root
CPU pyproject.toml and uv.lock remain unchanged. GPU Torch is exactly
2.8.0+cu128 from https://download.pytorch.org/whl/cu128. All non-Torch packages
shared with the CPU environment retain their pinned CPU-lock versions. Additional
CUDA libraries and Triton are pinned in this directory's uv.lock.

## Install from the repository root

```sh
uv sync --project environments/cuda128 --locked --python 3.11.16 --default-index https://pypi.org/simple
uv pip install --python environments/cuda128/.venv/bin/python --no-deps --no-sources --no-build-isolation -e .
uv pip check --python environments/cuda128/.venv/bin/python
environments/cuda128/.venv/bin/python environments/cuda128/verify_gpu.py
environments/cuda128/.venv/bin/pytest -q
```

Repeat the no-dependencies editable installation after each sync: sync removes
packages outside this environment lock. The source package is installed separately
to avoid importing the root project's CPU-only Torch source configuration. Do not
activate the CPU environment for GPU commands; use the explicit paths above.
`.venv/` directories are already ignored, including this nested environment.
Track this directory's pyproject.toml and uv.lock. Package downloads may total
several GB. The PyTorch wheel supplies the required CUDA user-space libraries.

The verifier runs the actual residual CNN forward and backward on random masked
arrays under the same deterministic runtime settings as training. It reads no ECG
records and outputs GPU identity, Torch/CUDA versions, compiled architectures,
peak allocated memory, lock hashes and installed package versions. It fails if
CUDA is unavailable; no CPU fallback is used. Save its output and the subsequent
smoke report under ignored outputs/ for run provenance.

## GPU smoke check

After the verifier and tests pass:

```sh
environments/cuda128/.venv/bin/rhythmlens smoke-rhythm   --config configs/rhythm-smoke-debug-cuda128-v1.json   --preprocessing outputs/rhythm/smoke-seed42-v1/preprocessing.json   --output outputs/rhythm/smoke-debug-cuda128-seed42-v1
```

This retains the four-record CPU diagnostic settings, changing only the device.
It reads no validation or holdout signals. Its results are training diagnostics.

## Full baseline

After GPU execution and the GPU smoke have been checked:

```sh
environments/cuda128/.venv/bin/rhythmlens train-rhythm   --config configs/rhythm-baseline-cuda128-v1.json   --preprocessing outputs/rhythm/smoke-seed42-v1/preprocessing.json   --output outputs/rhythm/baseline-cuda128-seed42-v1
```

This uses the original baseline settings (including dropout 0.1), changing only
the device to CUDA. Training starts with fresh weights. Interrupted CPU weights
are not resumed. Validation selects the checkpoint; holdout signals remain
untouched. Existing output directories are never overwritten.

The checkpoint's existing dependency_lock_sha256 field still identifies the
frozen root CPU lock. For GPU reproduction also retain the separate GPU lock hash
and full installed versions emitted by verify_gpu.py, and the GPU config hash and
runtime metadata in the run report. CPU and GPU results need not be bit-identical.

## Preparation evidence and remaining checks

Prepared with uv 0.12.23. The GPU lock resolved 73 entries. Offline stale-lock
checking and installation dry-run passed. The proposed Linux environment installs
72 locked distributions plus the separately installed editable RhythmLens.
Shared non-Torch versions match the CPU lock; root CPU lock SHA256 remains
4ff3ef044d9e70d26b8a5c0494478e30cbc3f73ac3ecd94c109e1dd1adf44c8d.
The verifier passed Ruff and a synthetic CPU logic check with CUDA reporting
calls stubbed. No CUDA packages were installed here, and no GPU execution,
real ECG training, GPU performance or GPU accuracy was verified here.

Official sources:
- https://pytorch.org/get-started/previous-versions/#v2-8-0
- https://pytorch.org/blog/pytorch-2-7/ (Blackwell / CUDA 12.8 support)
- https://docs.astral.sh/uv/guides/integration/pytorch/
