# YAML and the launcher

`config.yaml`, `configs/reference.yaml` and `configs/temporal.yaml` use strict
schema 2: duplicate keys, unknown keys, missing settings, escaping paths and invalid
combinations fail before execution. Keep paths repository-relative and secrets out
of YAML. Legacy schema-1 JSON loading remains supported, but new experiments use YAML.
The CLI default is `config.yaml`.

- `experiment`: distinct name and 32-bit seed.
- `environment.profile`: `cpu` or `cuda128`, selecting separate locks/interpreters.
- `runtime`: `device` (`cpu`, `cuda`, explicit `auto`), visible `gpu_index`, CPU threads.
- `model`: `rhythm-cnn-v1` reference or `rhythm-dilated-v1` candidate, widths, kernels,
  dropout. Candidate dilations are fixed by its version; a change requires a new version.
- `optimizer` / `training`: unchanged AdamW, learning rate, weight decay, batch size,
  epochs, patience, caching, validation macro-F1 selection.
- `smoke`: training records per class, batch size, step/time limits and diagnostic
  dropout; bounded to 200 steps and 1,200 seconds. A smoke is not performance evidence.
- `preprocessing`: supported method and optional reusable training-fitted state.
- `paths.output_root`: ignored outputs/, runs/ or checkpoints/ subdirectory.
- `execution`: explicit `resume` and `noncanonical_development` booleans.
- `evaluation`: current `profile`, `device`, `gpu_index`, CPU threads, independent of
  checkpoint training runtime. CPU evaluation of CUDA-trained weights is supported.

```sh
python3 scripts/manage.py show --config configs/reference.yaml
python3 scripts/manage.py train --config configs/reference.yaml --dry-run
python3 scripts/manage.py doctor --config configs/reference.yaml --smoke
```

To change GPU, edit `runtime.gpu_index` before setup/doctor/train. The index refers
to visible devices (including any caller-provided CUDA visibility restriction).
To evaluate on CPU, set `evaluation.profile: cpu` and `evaluation.device: cpu`,
ensure the CPU environment was set up, then:

```sh
python3 scripts/manage.py evaluate --config config.yaml
```

The launcher selects the evaluation interpreter, loads training provenance intact,
uses the current evaluation runtime, and writes its actual environment separately.
The checkpoint path is derived from the training profile's run directory. Evaluation
is validation-only; no holdout option exists. It cannot present smoke as validation.
Calling `evaluate-rhythm` without a runtime YAML defaults safely to CPU.

After interruption, preserve the same configuration and run:

```sh
python3 scripts/manage.py train --config configs/reference.yaml --resume
```

Alternatively set only `execution.resume: true`, commit that reviewed YAML action
change to retain a clean canonical source state, and repeat the `train` action. Changing this resume flag
and its YAML content hash is allowed; learning settings, model, preprocessing, data,
source hashes, dependency locks, packages and runtime must agree. Original YAML and
resolved training configuration are preserved. The latest resume preflight is saved
separately. Completed runs reject resume. To start a different experiment choose a
new name and set resume false. Resume replays any partial epoch from `last.pt`; it
does not continue from partially updated in-memory weights. CPU determinism is
regression-tested; CPU/GPU numerical equivalence is not promised.

`make` delegates to the same launcher (`CONFIG=configs/reference.yaml`). No full
training is performed by setup, doctor, verify-data, test, build or dry-run.
