# CUDA 12.8 environment

Use YAML plus the canonical launcher from the repository root:

```sh
python3 scripts/manage.py setup --config config.yaml
python3 scripts/manage.py doctor --config config.yaml
```

`config.yaml` selects this separate exact lock and `environments/cuda128/.venv`.
The root package is installed without dependencies or build isolation after sync,
so the root CPU Torch source cannot replace CUDA Torch. Hatchling and editables
are pinned and installed by this environment lock. Edit GPU index through YAML.
Doctor includes real synthetic forward/backward and architecture/memory diagnostics;
`verify_gpu.py --config config.yaml` is a legacy alias. No ECG decoding occurs in
doctor. `doctor --smoke` permits development source and verifies training inputs.

See [reproduction](../../docs/reproduction.md) and
[configuration/resume](../../docs/configuration.md) for training, evaluation on CPU,
canonical source gates and epoch-boundary resume. Setup alone does not verify GPU
hardware. Current observed hardware evidence is in
[readiness](../../reports/pretraining-readiness.md). The prior instructions are
preserved in docs/archive/cuda128-before-readiness.md as historical context.
