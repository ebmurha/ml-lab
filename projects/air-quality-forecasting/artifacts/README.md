# Artifacts

Generated models and evaluation artifacts belong in this directory and are ignored by Git.

`air-quality-train` creates:

- `forecast_model.joblib`: the selected estimator, feature contract and deterministic model version.
- `metrics.json`: data quality, validation, holdout, reproducibility and memory results.

Artifact versions combine the selected candidate name with the source-data SHA-256 prefix. Re-running the fixed configuration against identical data therefore retains the same version identifier.
