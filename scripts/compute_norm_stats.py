"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon,
        model_config,
        load_images=False,
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def normalized_action_metrics(
    data_loader: _data_loader.Dataset,
    num_batches: int,
    action_stats: normalize.NormStats,
    *,
    use_quantiles: bool,
) -> tuple[float, float]:
    """Measure the exact normalized action scale over the source dataset."""

    max_abs = 0.0
    sum_squares = 0.0
    count = 0
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Validating normalized actions"):
        actions = np.asarray(batch["actions"], dtype=np.float64)
        if use_quantiles:
            assert action_stats.q01 is not None
            assert action_stats.q99 is not None
            normalized = (
                (actions - action_stats.q01)
                / (action_stats.q99 - action_stats.q01 + 1e-6)
                * 2.0
                - 1.0
            )
        else:
            normalized = (actions - action_stats.mean) / (action_stats.std + 1e-6)
        max_abs = max(max_abs, float(np.max(np.abs(normalized), initial=0.0)))
        sum_squares += float(np.sum(np.square(normalized), dtype=np.float64))
        count += int(normalized.size)
    if count == 0:
        raise ValueError("Cannot validate normalized actions from an empty dataset")
    return max_abs, sum_squares / count


def main(
    config_name: str,
    max_frames: int | None = None,
    num_workers: int | None = None,
    max_normalized_action_abs: float | None = None,
    max_normalized_action_mse: float | None = None,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers if num_workers is None else num_workers,
            max_frames,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    if max_normalized_action_abs is not None or max_normalized_action_mse is not None:
        action_max_abs, action_mse = normalized_action_metrics(
            data_loader,
            num_batches,
            norm_stats["actions"],
            use_quantiles=data_config.use_quantile_norm,
        )
        print(
            "Normalized action audit: "
            f"max_abs={action_max_abs:.6g}, mean_square={action_mse:.6g}"
        )
        if (
            max_normalized_action_abs is not None
            and action_max_abs > max_normalized_action_abs
        ):
            raise ValueError(
                f"Normalized action max_abs {action_max_abs:.6g} exceeds "
                f"limit {max_normalized_action_abs:.6g}"
            )
        if max_normalized_action_mse is not None and action_mse > max_normalized_action_mse:
            raise ValueError(
                f"Normalized action mean_square {action_mse:.6g} exceeds "
                f"limit {max_normalized_action_mse:.6g}"
            )

    output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
