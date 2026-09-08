import pandas as pd

from air_quality_forecasting.data import load_hourly_measurements


def test_loader_preserves_missing_hour_without_future_fill(raw_csv) -> None:
    source = pd.read_csv(raw_csv)
    missing_time = source.loc[10, "datetime_utc_start"]
    source = source[
        ~(
            source["parameter"].eq("pm25")
            & source["datetime_utc_start"].eq(missing_time)
        )
    ]
    source.to_csv(raw_csv, index=False)

    hourly, profile = load_hourly_measurements(raw_csv)

    assert pd.isna(hourly.loc[pd.Timestamp(missing_time), "pm25"])
    assert profile.missing_hours == 1
