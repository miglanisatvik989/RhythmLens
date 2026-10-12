# CUDA 12.8 environment

Separate Linux x86_64 environment, including WSL. Its exact lock and root editable
installation without dependencies/build isolation preserve CUDA Torch.

Use the [canonical workflow](../../README.md) with a CUDA configuration:

```sh
python3 scripts/manage.py setup --config configs/temporal-cuda128.yaml
python3 scripts/manage.py setup-evaluation --config configs/temporal-cuda128.yaml
python3 scripts/manage.py doctor --config configs/temporal-cuda128.yaml
```

The evaluator may use CPU independently. Setup prepares environments; doctor
verifies actual GPU execution, capability, architectures, and memory.
Choose GPU settings in a new committed configuration.
`verify_gpu.py --config <file>` remains an alias to doctor.

See [configuration](../../docs/configuration.md), [reproduction](../../docs/reproduction.md),
and [dated hardware evidence](../../reports/pretraining-readiness.md).
