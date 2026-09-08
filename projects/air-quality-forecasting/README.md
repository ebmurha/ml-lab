# Air Quality Forecasting

## Goal

Forecast PM2.5 concentration six hours ahead for a configured location to support early air-quality alerts.

## Dataset and target

The working dataset contains hourly measurements retrieved from [OpenAQ](https://docs.openaq.org) for location `5199863`. Source, licence, attribution and coverage details are recorded in [`data/README.md`](data/README.md).

Data contributed by The Demography Project; data provided by [AirGradient](https://www.airgradient.com) via [OpenAQ](https://openaq.org). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The target is PM2.5 concentration six hours after prediction time. Inputs are leakage-safe PM2.5, temperature and humidity lags and rolling statistics, plus the forecast hour and weekday.

## Setup

Requires Python 3.11 or newer.

```powershell
cd projects/air-quality-forecasting
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and set the OpenAQ API key. The key remains local and is never written to a dataset or log.

## Fetch data

```powershell
air-quality-fetch `
  --location-id 5199863 `
  --datetime-from 2025-07-25T00:00:00Z `
  --datetime-to 2026-09-08T23:59:59Z
```

Use `--parameters pm25 temperature relativehumidity` to fetch only the modeling signals. The command discovers sensors, splits long ranges into requests no longer than one year, retrieves every page and removes boundary duplicates.

## Train and evaluate

```powershell
air-quality-train `
  --data data/openaq_location_5199863_2025-07-25_2026-09-08.csv
```

This creates ignored, reproducible artifacts under `artifacts/`.

## Produce a batch forecast

```powershell
air-quality-forecast `
  --data data/openaq_location_5199863_2025-07-25_2026-09-08.csv
```

The command prints timestamped JSON and appends operational metrics to `logs/forecasts.jsonl`. If the target observation is already present, it also records absolute prediction error.

To run the same command in Docker:

```powershell
docker build -t air-quality-forecasting .
New-Item -ItemType Directory -Force logs | Out-Null
docker run --rm `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/artifacts:/app/artifacts:ro" `
  -v "${PWD}/logs:/app/logs" `
  air-quality-forecasting `
  --data data/openaq_location_5199863_2025-07-25_2026-09-08.csv
```

## Implementation approach

The loader converts long-form sensor measurements into a regular hourly table without filling gaps. Features use only observations at or before prediction time. For a target at `t + 6h`, the 24-hour seasonal baseline uses PM2.5 observed at `t - 18h`.

Four expanding validation windows compare persistence, the seasonal baseline and `HistGradientBoostingRegressor`. A six-hour embargo prevents training targets from crossing into each later evaluation period. The latest 30 days form the final chronological holdout.

The exact selection criterion is the lowest mean validation MAE across the four windows, with candidate name as a deterministic tie-breaker. After final evaluation, the selected candidate is refit on the complete supervised dataset for deployment. Boosted trees suit this limited tabular dataset because they model nonlinear lag interactions, handle missing inputs and train efficiently on a CPU. An LSTM would add sequence construction, tuning and compute cost without enough data to justify that complexity.

Production modules own retrieval, preparation, features, evaluation, training and inference. The [`forecasting-analysis.ipynb`](notebooks/forecasting-analysis.ipynb) notebook imports those modules to present the experiment visually.

Machine-readable evidence is available in [`verification.json`](verification.json).

## Evaluation metrics

Mean absolute error (MAE), in µg/m³, is the primary metric.

| Candidate | Mean validation MAE | Windows won | Holdout MAE |
| --- | ---: | ---: | ---: |
| HistGradientBoostingRegressor | 8.304 | 3 | 9.774 |
| 24-hour seasonal naive | 8.917 | 1 | 10.029 |
| Persistence | 11.473 | 0 | 12.577 |

## Verification results

- The selected model beat both baselines in three of four validation windows and on the final holdout.
- Repeated training with the fixed configuration reproduced predictions within `1e-12` absolute tolerance.
- Feature and split tests confirm that future observations and targets do not cross evaluation boundaries.
- Peak measured training memory was 190.55 MB, below the 6 GB limit.
- The batch command produced valid forecast JSON and logged freshness, missing-input rate, latency and model version.
- All nine automated tests passed, including the deployed boosted-tree inference path.

These thresholds are exercise acceptance criteria and do not establish production suitability.

## Limitations

- PM2.5 coverage is 89.96%, with 983 missing hours and a largest gap of 92 hours.
- Results describe one monitoring location and may not generalize to another location, sensor or season.
- The holdout improvement over the seasonal baseline is small and needs confirmation on newer and external data.
- Weather inputs come from the same station; no meteorological forecast, wind, rainfall, traffic or fire data is included.
- An air-quality forecast does not by itself establish the effect of any alert or intervention.

## Future improvements

- Validate across additional locations and later chronological holdouts.
- Compare strategies for long missing periods without introducing future information.
- Add external weather forecasts and relevant event data when reliably available at prediction time.
- Define operational alert thresholds from decision costs and public-health requirements.
- Aggregate freshness, missingness, latency and error logs into drift and reliability reports.
