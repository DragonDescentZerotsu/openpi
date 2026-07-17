import numpy as np
import pytest

from openpi.training import config as _config
from openpi.training import optimizer as _optimizer


def test_a1_white_noise_1k_schedule_and_config():
    config = _config.get_config("pi05_a1_piper_pipette_handoff_white_noise_1k_10k")

    assert config.data.repo_id == "aero_quest/piper_pipette_handoff_white_noise_1k"
    assert config.data.assets.asset_id == "aero_quest/piper_pipette_handoff_white_noise_1k_pose_state"
    assert config.num_train_steps == 10_000
    assert config.save_interval == 5_000
    assert config.keep_period == 5_000

    schedule = config.lr_schedule.create()
    values = {step: float(np.asarray(schedule(step))) for step in (0, 499, 500, 2_999, 3_000, 5_999, 6_000, 9_999)}
    assert values[0] < values[499] < 1e-4
    assert values[500] == pytest.approx(1e-4)
    assert values[2_999] == pytest.approx(1e-4)
    assert values[3_000] == pytest.approx(5e-5)
    assert values[5_999] == pytest.approx(5e-5)
    assert values[6_000] == pytest.approx(2e-5)
    assert values[9_999] == pytest.approx(2e-5)


def test_a2_tip_attachment_schedule_and_config():
    config = _config.get_config("pi05_a2_piper_aero_tip_attachment_10k")

    assert config.data.repo_id == "aero_quest/piper_aero_tip_attachment"
    assert config.data.assets.asset_id == "aero_quest/piper_aero_tip_attachment_vision_only"
    assert config.model.discrete_state_input is False
    assert config.num_train_steps == 10_000
    assert config.save_interval == 5_000
    assert config.keep_period == 5_000

    schedule = config.lr_schedule.create()
    values = {step: float(np.asarray(schedule(step))) for step in (0, 499, 500, 2_999, 3_000, 5_999, 6_000, 9_999)}
    assert values[0] < values[499] < 1e-4
    assert values[500] == pytest.approx(1e-4)
    assert values[2_999] == pytest.approx(1e-4)
    assert values[3_000] == pytest.approx(5e-5)
    assert values[5_999] == pytest.approx(5e-5)
    assert values[6_000] == pytest.approx(2e-5)
    assert values[9_999] == pytest.approx(2e-5)


def test_second_lr_drop_requires_complete_ordered_pair():
    with pytest.raises(ValueError, match="must be set together"):
        _optimizer.WarmupThenStepSchedule(second_drop_step=6_000).create()
    with pytest.raises(ValueError, match="greater than or equal to drop_step"):
        _optimizer.WarmupThenStepSchedule(second_drop_step=2_000, second_final_lr=2e-5).create()
