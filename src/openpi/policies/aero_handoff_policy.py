import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


ACTION_DIM = 20

# A1 policy action/state schema:
#   0:6   piper_original arm next-target offsets from observed qpos
#   6     piper_original gripper opening
#   7:13  piper_aerohand arm next-target offsets from observed qpos
#   13:20 semantic Aero Hand state/action in [0, 1]
# The full exported qpos remains 40-D; pi0.5 pads this 20-D state/action to 32-D.
ARM_DELTA_MASK = np.asarray([True] * 6 + [False] + [True] * 6 + [False] * 7, dtype=bool)
AERO_HAND_QPOS_GROUPS = (
    (26,),  # thumb_abduction
    (27,),  # thumb_flexion_1
    (28, 29),  # thumb_flexion_2
    (14, 15, 16),  # index_curl
    (17, 18, 19),  # middle_curl
    (20, 21, 22),  # ring_curl
    (23, 24, 25),  # pinky_curl
)
AERO_HAND_QPOS_RANGES = {
    14: (0.0, 1.5708),
    15: (0.0, 1.5708),
    16: (0.0, 1.5708),
    17: (0.0, 1.5708),
    18: (0.0, 1.5708),
    19: (0.0, 1.5708),
    20: (0.0, 1.5708),
    21: (0.0, 1.5708),
    22: (0.0, 1.5708),
    23: (0.0, 1.5708),
    24: (0.0, 1.5708),
    25: (0.0, 1.5708),
    26: (0.0, 1.7453),
    27: (0.0, 0.9559),
    28: (0.0, 1.5708),
    29: (0.0, 1.5708),
}


def make_aero_handoff_example() -> dict:
    return {
        "observation/state": np.random.rand(40).astype(np.float32),
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


def a1_state_from_qpos(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32)
    left = [*qpos[:6].tolist(), float(qpos[6])]
    right = qpos[8:14].tolist()
    hand = []
    for group in AERO_HAND_QPOS_GROUPS:
        values = []
        for index in group:
            lo, hi = AERO_HAND_QPOS_RANGES[index]
            values.append(float(np.clip((float(qpos[index]) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)))
        hand.append(float(np.mean(values)))
    return np.asarray([*left, *right, *hand], dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class AeroHandoffInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        table_image = _parse_image(data["observation/images/table_overview"])
        gripper_image = _parse_image(data["observation/images/gripper_forward"])
        palm_image = _parse_image(data["observation/images/palm_inner"])
        state = a1_state_from_qpos(np.asarray(data["observation/state"], dtype=np.float32))

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
