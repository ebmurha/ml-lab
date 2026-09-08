# Artifacts

Generated models and evaluation artifacts belong in this directory and are ignored by Git.

`air-quality-train` creates:

- `forecast_model.joblib`: the selected estimator, feature contract and deterministic model version.
- `metrics.json`: data quality, validation, holdout, reproducibility and memory results.

Artifact versions combine the selected candidate name with SHA-256 prefixes for the dataset, training configuration and normalized package source. Re-running identical code and configuration against identical data retains the same identifier, while changing any of those inputs creates a new version.
