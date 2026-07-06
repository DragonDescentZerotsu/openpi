# openpi baseline instructions

This directory is a git submodule for the openpi baseline. Keep project-wide
training glue in the parent SO_AeroHand repository unless the change is truly
openpi-specific.

## Environment

Do not use the project `aero_sim` environment for openpi training or pi0.5
evaluation. `aero_sim` is pinned for simulation and LeRobot export with Python
3.13 and NumPy 2.x, while this openpi checkout is locked for Python 3.11 and
NumPy 1.26.

Use the dedicated conda environment:

```bash
conda activate openpi_pi05
cd /data1/tianang/Projects/SO_AeroHand/baselines/openpi
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
```

`UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"` is required when using `uv` from this
checkout so dependencies are synchronized into the active conda environment
instead of a local `.venv`.

Check the installed environment with:

```bash
uv sync --frozen --check
python -c "import jax, torch; print(jax.devices()); print(torch.cuda.is_available(), torch.cuda.device_count())"
```

The validated baseline environment is:

- Python `3.11`
- NumPy `1.26.4`
- JAX/JAXLIB `0.5.3` with CUDA devices visible
- Torch `2.7.1`
- Transformers `4.53.2`
- Flax `0.10.2`
- Orbax Checkpoint `0.11.13`

## pi0.5 training and evaluation

For pi0.5, use openpi's flow-matching head support. Useful built-in configs
include `pi05_libero`, `pi05_droid`, and `debug_pi05`; project-specific A1
handoff configs should be added deliberately once the A1 train/eval dataset
schema is fixed.

Typical entry points are:

```bash
python scripts/compute_norm_stats.py --config-name <config_name>
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py <config_name> --exp-name=<run_name> --overwrite
python scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_dir>
```

Do not commit checkpoints, WandB logs, downloaded model assets, dataset caches,
or generated training outputs from this directory.

## Repository boundary

This submodule should contain openpi-specific model code, configs, and patches.
Cross-baseline or SO_AeroHand-specific code belongs in the parent repository,
using these locations unless there is a strong reason not to:

- `aero_train/` for shared dataset adapters, rollout/eval code, metrics, and
  training utilities.
- `scripts/training/` for project-level train/eval/prepare entry points.
- `configs/training/` for project-level experiment configs.

If modifying files inside this submodule, commit and push in the openpi fork
first, then update the parent SO_AeroHand submodule pointer in a separate parent
repository commit. Do not copy openpi source files into the parent repository to
avoid submodule workflow.
