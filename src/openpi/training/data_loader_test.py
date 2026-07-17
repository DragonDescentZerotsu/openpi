import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _write_aero_handoff_shard(root: Path, *, file_index: int, episode_index: int) -> None:
    rows = 2
    frame_index = np.arange(rows, dtype=np.int64)
    table = pd.DataFrame(
        {
            "observation.state": [np.full(20, episode_index, dtype=np.float32) for _ in range(rows)],
            "action": [np.zeros(20, dtype=np.float32) for _ in range(rows)],
            "controller.arm_qpos": [np.zeros(12, dtype=np.float32) for _ in range(rows)],
            "observation.stage_index": np.zeros(rows, dtype=np.int64),
            "timestamp": frame_index.astype(np.float32) / 30.0,
            "frame_index": frame_index,
            "episode_index": np.full(rows, episode_index, dtype=np.int64),
            "index": file_index * rows + frame_index,
            "task_index": np.zeros(rows, dtype=np.int64),
        }
    )
    data_path = root / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(data_path, index=False)


def test_aero_handoff_dataset_supports_multiple_data_and_video_shards(tmp_path: Path):
    for file_index in range(2):
        _write_aero_handoff_shard(
            tmp_path,
            file_index=file_index,
            episode_index=file_index,
        )

    episode_rows = {"episode_index": [0, 1]}
    for key in _data_loader.AERO_DUAL_PIPER_VIDEO_KEYS:
        prefix = f"videos/{key}"
        episode_rows[f"{prefix}/chunk_index"] = [0, 0]
        episode_rows[f"{prefix}/file_index"] = [0, 1]
        episode_rows[f"{prefix}/from_timestamp"] = [0.0, 0.5]
        for file_index in range(2):
            video_path = tmp_path / "videos" / key / "chunk-000" / f"file-{file_index:03d}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.touch()

    episode_meta_path = tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episode_meta_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_rows).to_parquet(episode_meta_path, index=False)
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"fps": 30}), encoding="utf-8")

    dataset = _data_loader.AeroDualPiperDataset(tmp_path, action_horizon=3)

    assert len(dataset) == 4
    np.testing.assert_array_equal(dataset._policy_states[:, 0], [0, 0, 1, 1])  # noqa: SLF001
    assert dataset._episode_bounds == {0: (0, 2), 1: (2, 4)}  # noqa: SLF001
    for key in _data_loader.AERO_DUAL_PIPER_VIDEO_KEYS:
        assert dataset._episode_video_refs[key][0][0].name == "file-000.mp4"  # noqa: SLF001
        assert dataset._episode_video_refs[key][1][0].name == "file-001.mp4"  # noqa: SLF001
        assert dataset._episode_video_refs[key][1][1] == 15  # noqa: SLF001
    assert dataset._action_chunk(0).shape == (3, 20)  # noqa: SLF001

    stats_dataset = _data_loader.AeroDualPiperDataset(
        tmp_path,
        action_horizon=3,
        load_images=False,
    )
    for key in _data_loader.AERO_DUAL_PIPER_VIDEO_KEYS:
        assert stats_dataset._read_image(key, 0).shape == (1, 1, 3)  # noqa: SLF001


def test_a2_dataset_uses_task_specific_root_and_prompt():
    root, prompt = _data_loader.AERO_DUAL_PIPER_DATASETS[_data_loader.AERO_TIP_ATTACHMENT_REPO_ID]

    assert root.name == "a2_well_holdout_train760_eval40_v0"
    assert "next available tip" in prompt
    assert prompt != _data_loader.AERO_HANDOFF_PROMPT


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
