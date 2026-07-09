import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 20
STATE_DIM = 20
STATE_MODE_POSE = "pose"
STATE_MODE_JOINT = "joint"
STATE_MODES = (STATE_MODE_POSE, STATE_MODE_JOINT)

# A1 policy state schema:
#   pose mode:
#     0:6   left original Piper eef pose in the left robot base frame
#     6     left original gripper opening
#     7:13  right Piper + Aero Hand palm pose in the right robot base frame
#     13:20 semantic Aero Hand state in [0, 1]
#   joint mode:
#     0:6   left original Piper arm qpos
#     6     left original gripper opening
#     7:13  right Piper + Aero Hand arm qpos
#     13:20 semantic Aero Hand state in [0, 1]
# A1 policy action schema:
#   0:6   left arm next-target offsets from controller measured qpos
#   6     left original gripper absolute target
#   7:13  right arm next-target offsets from controller measured qpos
#   13:20 semantic Aero Hand absolute target in [0, 1]
# Full MuJoCo qpos and object/environment state must not be exposed through
# observation.state. The dataset's controller.arm_qpos field is a controller-only
# helper for chunk rebasing and is not passed to the model.
ARM_DELTA_MASK = np.asarray([True] * 6 + [False] + [True] * 6 + [False] * 7, dtype=bool)


def validate_state_mode(state_mode: str) -> str:
    if state_mode not in STATE_MODES:
        raise ValueError(f"Unknown A1 state mode {state_mode!r}; expected one of {STATE_MODES}")
    return state_mode


def make_aero_handoff_example() -> dict:
    return {
        "observation/state": np.random.rand(STATE_DIM).astype(np.float32),
        "observation/images/table_overview": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/gripper_forward": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/palm_inner": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "handoff the pipette",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def validate_a1_policy_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"Expected {STATE_DIM}D A1 robot-only policy state, got {state.shape}")
    return state


@dataclasses.dataclass(frozen=True)
class AeroHandoffInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        table_image = _parse_image(data["observation/images/table_overview"])
        gripper_image = _parse_image(data["observation/images/gripper_forward"])
        palm_image = _parse_image(data["observation/images/palm_inner"])
        state = validate_a1_policy_state(np.asarray(data["observation/state"], dtype=np.float32))

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": table_image,
                "left_wrist_0_rgb": gripper_image,
                "right_wrist_0_rgb": palm_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class AeroHandoffOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
