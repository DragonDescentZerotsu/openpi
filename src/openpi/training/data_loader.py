from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.policies.aero_handoff_policy as aero_handoff_policy
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


AERO_HANDOFF_REPO_ID = "aero_quest/piper_pipette_handoff"
AERO_HANDOFF_WHITE_NOISE_1K_REPO_ID = "aero_quest/piper_pipette_handoff_white_noise_1k"
AERO_TIP_ATTACHMENT_REPO_ID = "aero_quest/piper_aero_tip_attachment"
AERO_TIP_ATTACHMENT_MIXED_V1_REPO_ID = "aero_quest/piper_aero_tip_attachment_mixed_v1"
SO_AEROHAND_ROOT = pathlib.Path(__file__).resolve().parents[5]
AERO_HANDOFF_PROMPT = (
    "Use the original Piper gripper to pick a pipette from the rack, hand it to the Aero Hand palm, "
    "and close four non-thumb fingers to hold the pipette."
)
AERO_TIP_ATTACHMENT_PROMPT = (
    "Insert the carried pipette into the next available tip in column-major order, attach it, "
    "and lift it clear of the tip box."
)
AERO_DUAL_PIPER_DATASETS = {
    AERO_HANDOFF_REPO_ID: (
        SO_AEROHAND_ROOT
        / "outputs/lerobot/piper_pipette_handoff/a1_libero_like_train50_eval20_320x320_30fps",
        AERO_HANDOFF_PROMPT,
    ),
    AERO_HANDOFF_WHITE_NOISE_1K_REPO_ID: (
        SO_AEROHAND_ROOT
        / "outputs/lerobot/piper_pipette_handoff/"
        "a1_white_noise_train1000_250clean_750perturbed_320x320_30fps_v1",
        AERO_HANDOFF_PROMPT,
    ),
    AERO_TIP_ATTACHMENT_REPO_ID: (
        SO_AEROHAND_ROOT
        / "outputs/lerobot/piper_aero_tip_attachment/a2_well_holdout_train760_eval40_v0",
        AERO_TIP_ATTACHMENT_PROMPT,
    ),
    AERO_TIP_ATTACHMENT_MIXED_V1_REPO_ID: (
        SO_AEROHAND_ROOT
        / "outputs/lerobot/piper_aero_tip_attachment/"
        "a2_well_holdout_train760_clean152_perturbed_eval40_v1",
        AERO_TIP_ATTACHMENT_PROMPT,
    ),
}
AERO_DUAL_PIPER_VIDEO_KEYS = (
    "observation.images.table_overview",
    "observation.images.gripper_forward",
    "observation.images.palm_inner",
)
# Historical public name retained for older project-side tests and scripts.
AERO_HANDOFF_VIDEO_KEYS = AERO_DUAL_PIPER_VIDEO_KEYS
IMAGE_CACHE_SIZE = 224


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class AeroDualPiperDataset(Dataset):
    """Local reader for generated dual-Piper LeRobot v3 datasets.

    The openpi-pinned LeRobot loader expects older per-episode metadata files.
    The generated SO_AeroHand dataset is a compact v3 chunked parquet/video
    export, so we read it directly here while preserving the same sample keys
    expected by openpi transforms.
    """

    def __init__(
        self,
        root: pathlib.Path,
        action_horizon: int,
        *,
        state_mode: str = aero_handoff_policy.STATE_MODE_POSE,
        load_images: bool = True,
        prompt: str = AERO_HANDOFF_PROMPT,
    ):
        import pandas as pd

        self._root = root
        self._action_horizon = action_horizon
        self._state_mode = aero_handoff_policy.validate_state_mode(state_mode)
        self._load_images = bool(load_images)
        self._prompt = str(prompt)
        parquet_files = sorted((root / "data").glob("**/*.parquet"))
        if not parquet_files:
            raise ValueError(f"No LeRobot parquet files found under {root / 'data'}")
        data_columns = [
            "observation.state",
            "action",
            "controller.arm_qpos",
            "observation.stage_index",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]
        table = pd.concat(
            [pd.read_parquet(path, columns=data_columns) for path in parquet_files],
            ignore_index=True,
        )
        source_policy_states = np.stack(table["observation.state"].map(np.asarray).to_numpy()).astype(np.float32)
        if source_policy_states.ndim != 2 or source_policy_states.shape[1] != aero_handoff_policy.STATE_DIM:
            raise ValueError(
                f"Expected dual-Piper source observation.state width {aero_handoff_policy.STATE_DIM}, "
                f"got {source_policy_states.shape}"
            )
        raw_actions = np.stack(table["action"].map(np.asarray).to_numpy()).astype(np.float32)
        if raw_actions.ndim != 2 or raw_actions.shape[1] != aero_handoff_policy.ACTION_DIM:
            raise ValueError(
                f"Expected dual-Piper action width {aero_handoff_policy.ACTION_DIM}, got {raw_actions.shape}"
            )
        self._actions = raw_actions
        if "controller.arm_qpos" not in table.columns:
            raise ValueError(
                "Dual-Piper pose-state datasets must include controller.arm_qpos for action chunk rebasing"
            )
        self._controller_arm_qpos = np.stack(table["controller.arm_qpos"].map(np.asarray).to_numpy()).astype(np.float32)
        expected_controller_dim = int(aero_handoff_policy.ARM_DELTA_MASK.sum())
        if self._controller_arm_qpos.ndim != 2 or self._controller_arm_qpos.shape[1] != expected_controller_dim:
            raise ValueError(
                f"Expected dual-Piper controller.arm_qpos width {expected_controller_dim}, "
                f"got {self._controller_arm_qpos.shape}"
            )
        if self._state_mode == aero_handoff_policy.STATE_MODE_POSE:
            self._policy_states = source_policy_states
        else:
            self._policy_states = np.concatenate(
                [
                    self._controller_arm_qpos[:, :6],
                    source_policy_states[:, 6:7],
                    self._controller_arm_qpos[:, 6:12],
                    source_policy_states[:, 13:20],
                ],
                axis=1,
            ).astype(np.float32)
        if self._policy_states.shape != source_policy_states.shape:
            raise ValueError(f"Bad dual-Piper {self._state_mode} state shape {self._policy_states.shape}")
        self._stage_index = table["observation.stage_index"].to_numpy(dtype=np.int64)
        self._timestamp = table["timestamp"].to_numpy(dtype=np.float32)
        self._frame_index = table["frame_index"].to_numpy(dtype=np.int64)
        self._episode_index = table["episode_index"].to_numpy(dtype=np.int64)
        self._index = table["index"].to_numpy(dtype=np.int64)
        self._task_index = table["task_index"].to_numpy(dtype=np.int64)
        episode_starts = np.flatnonzero(np.r_[True, self._episode_index[1:] != self._episode_index[:-1]])
        episode_ends = np.r_[episode_starts[1:], len(self._episode_index)]
        self._episode_bounds: dict[int, tuple[int, int]] = {}
        for start, end in zip(episode_starts, episode_ends, strict=True):
            episode_index = int(self._episode_index[start])
            if episode_index in self._episode_bounds:
                raise ValueError(
                    f"Dual-Piper episode {episode_index} is not contiguous in the data parquet files"
                )
            expected_frames = np.arange(end - start, dtype=np.int64)
            if not np.array_equal(self._frame_index[start:end], expected_frames):
                raise ValueError(
                    f"Dual-Piper episode {episode_index} has non-contiguous frame_index values"
                )
            self._episode_bounds[episode_index] = (int(start), int(end))
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        self._fps = int(info["fps"])
        episode_meta_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
        if not episode_meta_files:
            raise ValueError(f"No dual-Piper episode metadata found under {root / 'meta' / 'episodes'}")
        episode_meta_columns = ["episode_index"]
        for key in AERO_DUAL_PIPER_VIDEO_KEYS:
            prefix = f"videos/{key}"
            episode_meta_columns.extend(
                [
                    f"{prefix}/chunk_index",
                    f"{prefix}/file_index",
                    f"{prefix}/from_timestamp",
                ]
            )
        episode_meta = pd.concat(
            [pd.read_parquet(path, columns=episode_meta_columns) for path in episode_meta_files],
            ignore_index=True,
        )
        self._episode_video_refs: dict[str, dict[int, tuple[pathlib.Path, int]]] = {}
        for key in AERO_DUAL_PIPER_VIDEO_KEYS:
            prefix = f"videos/{key}"
            required = (
                "episode_index",
                f"{prefix}/chunk_index",
                f"{prefix}/file_index",
                f"{prefix}/from_timestamp",
            )
            missing_columns = [column for column in required if column not in episode_meta.columns]
            if missing_columns:
                raise ValueError(f"Missing dual-Piper episode video metadata columns: {missing_columns}")
            refs: dict[int, tuple[pathlib.Path, int]] = {}
            for row in episode_meta.loc[:, list(required)].itertuples(index=False, name=None):
                episode_index, chunk_index, file_index, from_timestamp = row
                path = root / "videos" / key / f"chunk-{int(chunk_index):03d}" / f"file-{int(file_index):03d}.mp4"
                refs[int(episode_index)] = (
                    path,
                    round(float(from_timestamp) * self._fps),
                )
            self._episode_video_refs[key] = refs
        missing = sorted(
            {
                str(path)
                for refs in self._episode_video_refs.values()
                for path, _start_frame in refs.values()
                if not path.is_file()
            }
        )
        if missing:
            raise FileNotFoundError(f"Missing dual-Piper video files: {missing}")
        self._captures: dict[pathlib.Path, object] = {}
        self._image_arrays = None
        self._image_cache_paths = {
            key: root / "openpi_cache" / f"{key.replace('.', '_')}_{IMAGE_CACHE_SIZE}.npy"
            for key in AERO_DUAL_PIPER_VIDEO_KEYS
        }

    def __len__(self) -> int:
        return int(self._policy_states.shape[0])

    def _get_capture(self, path: pathlib.Path):
        if path not in self._captures:
            import cv2

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video {path}")
            self._captures[path] = cap
        return self._captures[path]

    def _get_image_arrays(self):
        if self._image_arrays is None:
            if all(path.is_file() for path in self._image_cache_paths.values()):
                self._image_arrays = {
                    key: np.load(path, mmap_mode="r") for key, path in self._image_cache_paths.items()
                }
            else:
                self._image_arrays = {}
        return self._image_arrays

    def _read_image(self, key: str, index: int) -> np.ndarray:
        if not self._load_images:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        image_arrays = self._get_image_arrays()
        if key in image_arrays:
            return np.asarray(image_arrays[key][index])

        import cv2

        episode_index = int(self._episode_index[index])
        try:
            path, episode_start_frame = self._episode_video_refs[key][episode_index]
        except KeyError as exc:
            raise RuntimeError(f"Missing video metadata for {key} episode {episode_index}") from exc
        video_frame = episode_start_frame + int(self._frame_index[index])
        cap = self._get_capture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read {key} dataset frame {index} from {path} frame {video_frame}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _action_chunk(self, index: int) -> np.ndarray:
        ep = int(self._episode_index[index])
        episode_start, episode_end = self._episode_bounds[ep]
        local = int(self._frame_index[index])
        query_local = np.minimum(
            local + np.arange(self._action_horizon),
            episode_end - episode_start - 1,
        )
        query_indices = episode_start + query_local
        chunk = self._actions[query_indices].copy()
        mask = aero_handoff_policy.ARM_DELTA_MASK
        # Stored dual-Piper arm actions are per-frame next-target offsets
        # (target[k+1] - qpos[k]). A pi0/pi0.5 action chunk expects every arm
        # delta in the chunk to be relative to the controller qpos at the chunk
        # start, so convert to target[k+1] - qpos[index] here. The policy state
        # itself is FK pose features and is never used for this rebasing.
        chunk[:, mask] += self._controller_arm_qpos[query_indices] - self._controller_arm_qpos[index]
        return chunk

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = int(index.__index__())
        return {
            "observation.state": self._policy_states[idx],
            "action": self._action_chunk(idx),
            "observation.stage_index": np.asarray([self._stage_index[idx]], dtype=np.int64),
            "observation.images.table_overview": self._read_image("observation.images.table_overview", idx),
            "observation.images.gripper_forward": self._read_image("observation.images.gripper_forward", idx),
            "observation.images.palm_inner": self._read_image("observation.images.palm_inner", idx),
            "timestamp": np.asarray([self._timestamp[idx]], dtype=np.float32),
            "frame_index": np.asarray([self._frame_index[idx]], dtype=np.int64),
            "episode_index": np.asarray([self._episode_index[idx]], dtype=np.int64),
            "index": np.asarray([self._index[idx]], dtype=np.int64),
            "task_index": np.asarray([self._task_index[idx]], dtype=np.int64),
            "prompt": self._prompt,
        }


# Backward-compatible import name for project code written before A2 reuse.
AeroHandoffDataset = AeroDualPiperDataset


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    load_images: bool = True,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    if repo_id in AERO_DUAL_PIPER_DATASETS:
        root, prompt = AERO_DUAL_PIPER_DATASETS[repo_id]
        return AeroDualPiperDataset(
            root,
            action_horizon,
            state_mode=data_config.state_mode or aero_handoff_policy.STATE_MODE_POSE,
            load_images=load_images,
            prompt=prompt,
        )

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
