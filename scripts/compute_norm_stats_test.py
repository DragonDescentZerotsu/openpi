import numpy as np

from openpi.shared import normalize

from . import compute_norm_stats


def test_normalized_action_metrics_reports_quantile_outlier_scale() -> None:
    stats = normalize.NormStats(
        mean=np.array([0.0]),
        std=np.array([0.1]),
        q01=np.array([-0.01]),
        q99=np.array([0.01]),
    )
    loader = [
        {"actions": np.array([[[-0.01], [0.01]]])},
        {"actions": np.array([[[0.10]]])},
    ]

    max_abs, mean_square = compute_norm_stats.normalized_action_metrics(
        loader,
        num_batches=2,
        action_stats=stats,
        use_quantiles=True,
    )

    assert max_abs > 8.9
    assert mean_square > 27.0
