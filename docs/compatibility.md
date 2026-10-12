# Compatibility entry points

New experiments use schema-2 YAML and `scripts/manage.py`.
These interfaces remain for historical artifacts and scripts:

| Entry point | Retained purpose |
| --- | --- |
| `configs/rhythm-baseline-v1.json`, `configs/rhythm-baseline-cuda128-v1.json` | Legacy flat baseline examples |
| `configs/rhythm-smoke-debug-v1.json`, `configs/rhythm-smoke-debug-cuda128-v1.json` | Legacy diagnostics |
| Installed `train-rhythm`, `smoke-rhythm`, `evaluate-rhythm` CLI | Low-level commands used by the launcher |
| `environments/cuda128/verify_gpu.py --config <file>` | Alias to configured doctor |
| YAML `execution.resume` | Retained continuation action; prefer launcher `train --resume` |
| `fit-preprocessing` | Optional full-training state for explicit reuse |

Existing paths and contents are preserved. Moving compatibility files would
change source identity and break historical references. Known checkpoint versions
and legacy loading remain supported.

Audit/split JSON describes frozen data provenance. Original YAML and resolved JSON
run artifacts are complementary provenance, not competing new-training formats.
Historical replay/resume uses the recorded source revision and environment;
workflow changes do not rewrite checkpoints or weaken compatibility gates.
