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
cd baselines/openpi
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

Important pi0.5 state-input invariant: `Pi0Config(pi05=True,
discrete_state_input=False)` does **not** use the numerical values in
`observation.state`. `TokenizePrompt` omits state when
`discrete_state_input=False`, while the continuous `state_proj` suffix token is
only constructed and used by pi0 (`pi05=False`). The state tensor is still
loaded, normalized, padded, and carried in `Observation`; pi0.5 inference also
reads `observation.state.shape[0]` for the batch size, but changing the state
values cannot affect the training loss or predicted actions. The relevant
implementation paths are `src/openpi/transforms.py` in
`TokenizePrompt.__call__` and `src/openpi/models/pi0.py` in `Pi0.__init__`,
`embed_prefix`, and `embed_suffix`; check both paths when changing this behavior.
Therefore:

- `pi05=True, discrete_state_input=False` is an image + prompt (vision-only
  with respect to proprioception) ablation, not a continuous-state model.
- Setting `state_mode=pose` or `state_mode=joint`, computing state norm stats,
  or including `observation.state` in parquet does not make this configuration
  state-conditioned.
- To make pi0.5 consume robot state, use `discrete_state_input=True`; normalized
  state is then discretized into the prompt token stream. Use norm stats whose
  state schema matches the selected `state_mode`.
- Do not name new `discrete_state_input=False` pi0.5 experiments `pose_state`,
  `joint_state`, or `state20`. Use an explicit name such as `vision_only` or
  `image_prompt_ablation`. Historical config/run names may remain for artifact
  compatibility, but their documentation must identify them as state-blind.

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
  `pi05_a1_piper_pipette_handoff_pose_state`: historical image + prompt
  ablation with `discrete_state_input=False`. Despite the second config's
  legacy `pose_state` name and its 20D pose state data pipeline, the pi0.5 model
  does not consume those state values.
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
OPENPI_DATA_HOME="$HOME/.cache/openpi" \
JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax" \
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

The 1000-episode clean/recovery run uses the separate config
`pi05_a1_piper_pipette_handoff_white_noise_1k_10k`. It is pinned to local repo
id `aero_quest/piper_pipette_handoff_white_noise_1k`, dataset
`a1_white_noise_train1000_250clean_750perturbed_320x320_30fps_v1`, and asset id
`aero_quest/piper_pipette_handoff_white_noise_1k_pose_state`; do not point the
historical 50-demo config at this dataset. This config also has
`pi05=True, discrete_state_input=False`, so it is state-blind even though its
asset id and historical run name contain `pose_state`; it trains on the three
camera views and prompt, not the 20D observation-state values. Its schedule is
warmup steps 0-499,
`1e-4` through step 2999, `5e-5` for steps 3000-5999, and `2e-5` from step 6000
through the 10k end. Global batch size is 512, so 10,000 steps over 1,141,373
frames is `5,120,000 / 1,141,373 = 4.485825405` frame-level epochs. With
`save_interval=5000`, JAX writes checkpoint 5000 and the final checkpoint 9999.

Compute this config's independent norm stats without video decoding or spawned
dataset copies:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n openpi_pi05 \
  python scripts/compute_norm_stats.py \
    --config-name pi05_a1_piper_pipette_handoff_white_noise_1k_10k \
    --num-workers 0
```

The output is under
`assets/pi05_a1_piper_pipette_handoff_white_noise_1k_10k/aero_quest/piper_pipette_handoff_white_noise_1k_pose_state/`.
`compute_norm_stats.py` deliberately passes `load_images=False`; state/action
stats do not depend on camera pixels, and decoding three videos per frame would
only waste time.

The A2 tip-attachment config is `pi05_a2_piper_aero_tip_attachment_10k`. It
uses the local `aero_quest/piper_aero_tip_attachment` dataset at
`outputs/lerobot/piper_aero_tip_attachment/a2_well_holdout_train760_eval40_v0`
through the same dual-Piper 20D state/action reader, but with its own task
prompt, asset id, norm stats, and checkpoint namespace. The dataset contains
760 train episodes and 278,185 frames; the 40 fixed ID/OOD eval initializations
remain outside the training parquet. This first A2 baseline intentionally
matches the last A1 architecture and is therefore a three-camera image + prompt
run with `discrete_state_input=False`; do not describe it as state-conditioned.
Its schedule is also identical to the last A1 10k run: 500-step warmup,
`1e-4` through step 2999, `5e-5` through step 5999, and `2e-5` thereafter.
Use global batch 512 on 8 GPUs and save at step 5000 plus final step 9999.

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n openpi_pi05 \
  python scripts/compute_norm_stats.py \
    --config-name pi05_a2_piper_aero_tip_attachment_10k \
    --num-workers 0 \
    --max-normalized-action-abs 10 \
    --max-normalized-action-mse 2

OPENPI_DATA_HOME="$HOME/.cache/openpi" \
JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
conda run --no-capture-output -n openpi_pi05 \
  python scripts/train.py pi05_a2_piper_aero_tip_attachment_10k \
    --exp-name a2_tip_attachment_train760_vision_only_lr1e4_drop3k5e5_drop6k2e5_b512_fsdp8_10k \
    --overwrite \
    --fsdp-devices 8
```

For A2, keep both normalized-action audit limits in the norm-stat command. The
audit reuses the exact training transform over a second image-free data pass and
fails before writing stats when `max_abs > 10` or normalized mean-square `> 2`.
Do not replace this check with action clipping: extreme values here indicate a
bad expert rollout (for example, a supposedly parked observation arm moving),
which must be removed and replanned.

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

Artifacts are under the parent repository path:

```text
outputs/openpi_eval/a1_piper_pipette_handoff
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
train rollout: outputs/openpi_eval/a1_piper_pipette_handoff/pi05_chunk_base_delta_b512_fsdp8_step0999_train_ep000000_replan5
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
