import pandas as pd

from air_quality_forecasting.features import (
    SEASONAL_COLUMN,
    TARGET_COLUMN,
    make_feature_frame,
    make_supervised_dataset,
)


def test_future_observations_do_not_change_existing_features(hourly_frame) -> None:
    cutoff = hourly_frame.index[250]
    expected = make_feature_frame(hourly_frame).loc[:cutoff]
    changed = hourly_frame.copy()
    changed.loc[changed.index > cutoff] = 100_000

    actual = make_feature_frame(changed).loc[:cutoff]

    pd.testing.assert_frame_equal(actual, expected)


def test_target_and_seasonal_baseline_use_correct_times(hourly_frame) -> None:
    dataset, _ = make_supervised_dataset(hourly_frame)
    timestamp = dataset.index[100]

    assert dataset.at[timestamp, TARGET_COLUMN] == hourly_frame.at[
        timestamp + pd.Timedelta("6h"), "pm25"
    ]
    assert dataset.at[timestamp, SEASONAL_COLUMN] == hourly_frame.at[
        timestamp - pd.Timedelta("18h"), "pm25"
    ]
