# Air Quality Forecasting

## Goal

Forecast PM2.5 concentration six hours ahead for a configured location to support early air-quality alerts.

## Dataset and target

The project will use a recent continuous 6–12 month period of hourly PM2.5 observations for a selected location from [OpenAQ](https://docs.openaq.org). The location, dataset selection, and licence details will be recorded in [`data/README.md`](data/README.md) after acquisition.

The target is PM2.5 concentration six hours after the forecast timestamp. Candidate predictors include leakage-safe PM2.5 lags and rolling statistics, temperature, humidity, hour, and weekday, subject to availability in the selected data.

## Setup

Requires Python 3.11 or newer.

```powershell
cd projects/air-quality-forecasting
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and set the OpenAQ API key. The key is loaded locally and is never written to the dataset.

## Fetch data

Fetch every sensor for a location across a UTC range:

```powershell
air-quality-fetch `
  --location-id 5199863 `
  --datetime-from 2025-07-25T00:00:00Z `
  --datetime-to 2026-09-08T23:59:59Z
```

Use `--parameters pm25 temperature relativehumidity` to retrieve only the variables required for the initial forecast. The command discovers matching sensors, splits long ranges into requests of at most one year, retrieves every page, removes boundary duplicates, and writes one CSV under `data/`.

## Implementation approach

The implementation will compare persistence and 24-hour seasonal-naive baselines with `HistGradientBoostingRegressor`. Evaluation will use expanding-window validation and a final chronological holdout.

Production code owns API retrieval and will own data loading, feature creation, training, evaluation, and batch inference. The notebook will import that code and present the experiment visually.

## Evaluation metrics

Primary evaluation will use mean absolute error (MAE). The trained model must beat both baselines on the final holdout and in most validation windows.

## Verification results

Pending data acquisition and implementation.

## Limitations

The available monitoring locations, measurement continuity, provider coverage, and environmental variables may limit forecast quality and generalization.

## Future improvements

To be determined from the initial data and evaluation results.
