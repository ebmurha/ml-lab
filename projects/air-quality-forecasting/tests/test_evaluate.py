from air_quality_forecasting.evaluate import expanding_windows, split_final_holdout
from air_quality_forecasting.features import make_supervised_dataset


def test_chronological_splits_embargo_future_targets(hourly_frame) -> None:
    dataset, _ = make_supervised_dataset(hourly_frame)
    development, holdout, holdout_start = split_final_holdout(dataset)

    assert development["target_timestamp_utc"].max() < holdout_start
    assert holdout.index.min() >= holdout_start
    for training, validation in expanding_windows(development):
        assert training["target_timestamp_utc"].max() < validation.index.min()
