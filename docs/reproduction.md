# Reproduction

## Locked setup

Use Python 3.11.16, uv 0.12.5, and Git. The interpreter pin is in
[.python-version](../.python-version). Obtain uv from its official distribution
and check `uv --version`. The launcher can bootstrap pinned PyYAML before setup.

Choose a [configuration](configuration.md#available-configurations):

```sh
CONFIG=configs/reference.yaml
python3 scripts/manage.py setup --config "$CONFIG"
python3 scripts/manage.py setup-evaluation --config "$CONFIG"
```

CPU syncs the root lock with all extras. CUDA syncs its separate lock, then installs
the root editable package with `--no-deps --no-sources --no-build-isolation`.
Both locks and build helpers are pinned. Setup does not silently upgrade packages.
CPU and CUDA environments coexist; root CPU Torch does not replace CUDA Torch.
Prepare evaluation separately when its profile differs. Doctor verifies actual
configured device execution; installation alone does not verify hardware.

## Restore data

Restore the official `training2017.zip` and `REFERENCE-v3.csv` from
[recorded provenance](../data/manifests/challenge2017-v3/provenance.json).
The acquisition command checks recorded hashes and restores the ignored layout.

After CPU setup (or CPU setup-evaluation), reuse local files:

```sh
.venv/bin/python -m rhythmlens acquire \
  --local-archive /path/to/training2017.zip \
  --local-reference /path/to/REFERENCE-v3.csv
```

Replace those paths. Alternatively, the explicit command
`.venv/bin/python -m rhythmlens acquire` downloads the official inputs.
If only CUDA is installed, its `environments/cuda128/.venv/bin/python` can run
the same command. Acquisition is an explicit action; verification never downloads.

Raw files live under `data/raw/challenge2017/`; the manifest lists required
MAT/header paths. Reuse existing manifests and splits. Do not audit or split
again, and do not alter recorded provenance.

```sh
python3 scripts/manage.py verify-data --config "$CONFIG"
python3 scripts/verify_frozen.py
```

Verification checks frozen metadata, accounting, compatibility, and train/validation
raw checksums. It decodes no ECG, executes no GPU work, creates no run output,
and never opens holdout signal files. It works with existing output directories.
Missing/corrupt inputs fail.

## Train and evaluate

From clean committed source and a new output location:

```sh
python3 scripts/manage.py doctor --config "$CONFIG"
python3 scripts/manage.py smoke --config "$CONFIG"
# Inspect the smoke overfit target and checkpoint reload before continuing.
python3 scripts/manage.py train --config "$CONFIG"
python3 scripts/manage.py evaluate --config "$CONFIG"
```

Smoke evaluates training examples only and is not generalization evidence.
Full training selects checkpoints using development validation macro F1.
Holdout evaluation stays locked.

`preprocessing.reuse: null` automatically fits the state on all frozen training
records. There is no required separate fit or intermediate YAML edit.
For repeated experiments, optional `fit-preprocessing` produces
`preprocessing.json`; a new committed YAML may explicitly reuse it.
The state must match the full training manifest and fitting IDs.

See [methods](methods.md) for numerical definitions and research limitations.

## Artifacts and interruption

Layout: `<output_root>/<experiment>/<training-profile>/seed<seed>/<action>/`.

Training saves original `experiment.yaml`, resolved `config.json`,
`preflight.json`, `preprocessing.json`, `status.json`, `history.json`,
atomic `best.pt`/`last.pt`, `run-report.json`, and validation artifacts.
Evaluation writes `evaluation-report.json`, predictions, and plots under the
training profile's evaluation path, even when inference uses CPU.

Existing directories cannot silently restart. Checkpoints, ECG, environments,
predictions, and outputs stay ignored; compact observed reports remain trackable.
A pre-first-epoch failure has no usable checkpoint: preserve that directory and
choose a new experiment. Hard termination may leave status running; the last
atomic checkpoint remains the completed boundary. See [resume](configuration.md#resume).

## Historical reproduction

Use the run's recorded Git revision, locks, original YAML, and preprocessing identity.
Create a fresh checkout/worktree at that revision, prepare its own environments,
restore the same raw files, and follow the recorded workflow. Do not copy a live
environment whose editable package points at another checkout.

Replace the placeholders:

```sh
git worktree add ../RhythmLens-replay RECORDED_COMMIT_SHA
cd ../RhythmLens-replay
```

Separate checkouts have separate ignored outputs, preserving completed artifacts.
Launcher/helper changes alter source-provenance hashes even when numerical code
is unchanged; historical resume uses the original revision. Compatibility checks
are not bypassed. A matching recipe does not promise bit-identical retraining
across different hardware or numerical environments.

## Frozen-data compatibility

Audit-time lock/source hashes remain historical provenance.
[data_compatibility.json](../src/rhythmlens/data_compatibility.json) separately
approves decoder semantics. Frozen hashes, labels, membership, duplicate separation,
accounting, and per-access raw checksums remain enforced. Active software is recorded
separately.

An intentional decoder/dependency change requires a reviewed contract/version,
synthetic physical-unit/shape/checksum/duplicate evidence, lock review, clean
installation checks, and tracked rationale. Never rewrite frozen metadata to
make a software change pass.

A new audited dataset/split version is a separate maintainer decision. Historical
audit/split commands cannot overwrite this release. Use a new version and verified
grouping identifiers; never invent patient identities.
