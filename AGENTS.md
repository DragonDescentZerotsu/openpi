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
- MuJoCo `3.10.0` for SO_AeroHand A1/A2 scene evaluation. The upstream
  `gym-aloha` dependency declares `mujoco<3`, but the SO_AeroHand generated
  task scenes use MuJoCo 3.x `<model>/<attach>` syntax and will not load with
  `mujoco==2.3.7`.

## pi0.5 training and evaluation

For pi0.5, use openpi's flow-matching head support. Useful built-in configs
include `pi05_libero`, `pi05_droid`, and `debug_pi05`. The project-specific A1
handoff config is `pi05_a1_piper_pipette_handoff`.

The SO_AeroHand A1 handoff configs use a 20D robot-only policy state and 20D
policy action schema, not full MuJoCo `qpos` or raw `model.nu`. There are two
supported state modes over the same generated LeRobot parquet:

- `pose`: left original Piper `link6` eef pose in the left robot base frame,
  left original gripper opening, right Piper + Aero Hand `palm` pose in the
  right robot base frame, and 7 semantic Aero Hand channels. Each pose is
  position xyz + axis-angle xyz and must be reproducible in real deployment from
  joint encoders, fixed robot model FK, and the robot's own base frame.
- `joint`: left original Piper arm qpos 6D, left original gripper opening, right
  Piper + Aero Hand arm qpos 6D, and 7 semantic Aero Hand channels. This mode is
  rebuilt in the OpenPI loader from the dataset's `controller.arm_qpos` helper
  plus gripper/hand fields, and intentionally has the same 20D ordering as the
  action schema.

Do not expose pipette freejoint pose, rack pose, pipette ejector/button/knob, or
other direct object/environment state through `observation.state`; these remain
raw-trajectory/debug fields only.
The 20D action order is left arm 6D, left gripper 1D, right arm 6D, and Aero
Hand 7D. The 12 arm entries are next-target offsets from controller measured
arm qpos. The stored LeRobot frame action is
`policy_ctrl[min(i + 1, T - 1)] - expert_qpos[i]` at the policy frame rate. In
perturbed raw traces, `ctrl` is the noisy command that generated the state
trajectory and `policy_ctrl` is the expert/recovery target used as the label;
legacy clean raw without `policy_ctrl` falls back to `ctrl`. The
dataset also contains a 12D `controller.arm_qpos` field used only by the OpenPI
A1 data loader to rebase action chunks to the chunk-start controller qpos; it is
not passed to the model. The parent evaluator executes the chunk closed-loop as
`ctrl = chunk_start_rollout_qpos + predicted_offset`, clipped to actuator range.
Do not add or revive a hidden command accumulator for A1 eval. The gripper and
Aero Hand semantic channels stay absolute targets from the same next target
frame. Passive pipette button/ejector actuators are part of the shared pipette
MJCF but are excluded from A1 pi0.5 training.

The A1 data loader is intentionally strict: source LeRobot `observation.state`
must be 20D robot-only pose proprioception, `controller.arm_qpos` must be 12D
arm encoder qpos for chunk rebasing and optional joint-state reconstruction, and
`action` must be the 20D A1 task action. Do not add compatibility conversion for
old full-qpos, 40D, or pre-20D parquet exports. Pose-state and joint-state
checkpoints/norm stats are not compatible even though the tensor shape is still
20D. The project policy transform pads state/actions from 20D to pi0.5's 32D
model action space; the padded tail must stay zero.

The local A1 reader supports both the historical single-file export and
no-concat aggregate datasets with multiple parquet/video shards. It concatenates
sorted data parquet files, requires contiguous episode/frame indices, and uses
`meta/episodes` video `chunk_index`, `file_index`, and `from_timestamp` fields to
resolve each frame. Do not restore hard-coded `file-000` video paths.

Use these A1 pi0.5 configs:

- `pi05_a1_piper_pipette_handoff` and
  `pi05_a1_piper_pipette_handoff_pose_state`: current pose-state version,
  `discrete_state_input=False`, matching the running pose-observation ablation.
- `pi05_a1_piper_pipette_handoff_joint_state_discrete`: joint-state version,
  `discrete_state_input=True`, so normalized 20D state is discretized into the
  pi0.5 prompt token stream. Compute separate norm stats for this config before
  training; it uses asset id `aero_quest/piper_pipette_handoff_joint_state`.

Typical entry points are:

```bash
python scripts/compute_norm_stats.py --config-name <config_name>
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py <config_name> --exp-name=<run_name> --overwrite
python scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_dir>
```

For A1 pi0.5 training, set the shared cache locations explicitly so downloaded
assets and JAX compilation cache do not land inside the submodule:

```bash
OPENPI_DATA_HOME=/data1/tianang/cache/openpi \
JAX_COMPILATION_CACHE_DIR=/data1/tianang/cache/jax \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run --no-capture-output -n openpi_pi05 \
  python scripts/train.py pi05_a1_piper_pipette_handoff \
    --exp-name <run_name> \
    --overwrite \
    --num-train-steps <steps> \
    --save-interval <steps> \
    --keep-period <steps> \
    --batch-size 512 \
    --fsdp-devices 8 \
    --num-workers 16
```

On 8x A100, `--batch-size 512 --fsdp-devices 8` uses all 8 GPUs at about
`64.8GB` per GPU for this full pi0.5 fine-tune. Avoid frequent checkpoints:
Orbax writes roughly `42GB` per saved step for params/train_state/assets, and
finalization can take several minutes even after the training loop is done.

The current `pi05_a1_piper_pipette_handoff` learning-rate schedule is a
global-step schedule: linearly warm up for 500 optimizer steps from
`1e-4 / 501` to `1e-4`, hold `1e-4` through step 2999, then switch to and hold
`5e-5` from step 3000 onward. The previous A1 setting warmed up for 500 steps
to constant `5e-5`.

The obsolete 2026-07-07 A1 command-delta debug run was:

```text
exp: a1_train50_eval20_cmd_delta_arm_b512_fsdp8_4k
checkpoint root: checkpoints/pi05_a1_piper_pipette_handoff/a1_train50_eval20_cmd_delta_arm_b512_fsdp8_4k
checkpoints: 999, step0999_preserved, 2000, 3000, 3999
wandb: https://wandb.ai/reasonv/openpi/runs/2q5ykmz8
loss: 0.6628 -> about 0.0082 over 4000 steps
```

The 1000-step checkpoint was successfully resumed to 4000 steps, but this run
used `ctrl_target[t] - ctrl_target[t-1]` labels plus a hidden command
accumulator in eval. Treat it as an invalid action-semantics debug artifact, not
as a corrected A1 baseline.

Closed-loop eval from the parent repo produced:

```text
step0999 eval_id: 0/20
step2000 eval_id: 0/20
step3000 eval_id: 0/20
step3999 eval_id: 0/20
step3999 train:   0/5
```

Artifacts are under:

```text
/data1/tianang/Projects/SO_AeroHand/outputs/openpi_eval/a1_piper_pipette_handoff
```

Treat these checkpoints as invalid action-semantics debug results, not solved A1
baselines. The next valid A1 pi0.5 runs should use next-target-offset labels,
fresh norm stats, and train-set rollout smoke tests before longer eval sweeps.

The first corrected chunk-base delta run was:

```text
exp: a1_train50_eval20_chunk_base_delta_b512_fsdp8_1k
checkpoint: checkpoints/pi05_a1_piper_pipette_handoff/a1_train50_eval20_chunk_base_delta_b512_fsdp8_1k/999
wandb: https://wandb.ai/reasonv/openpi/runs/ydbl8oim
loss: 0.6244 -> 0.0047 over 1000 steps
train rollout: /data1/tianang/Projects/SO_AeroHand/outputs/openpi_eval/a1_piper_pipette_handoff/pi05_chunk_base_delta_b512_fsdp8_step0999_train_ep000000_replan5
train success: 0/1
```

Do not extend that run to 4000 steps without further action-semantics work. The
seen-train rollout fails because tiny nonzero arm-delta predictions during the
initial hold window accumulate under current-qpos-relative execution; left-arm
ctrl diverges before pregrasp, so the pipette never leaves the rack.

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
