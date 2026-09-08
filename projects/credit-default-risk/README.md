# Credit Default Risk

## Goal

Predict the probability that an existing customer defaults on next month's credit-card payment so an outreach team can prioritize support. This model must not be used to approve, reject, or price credit.

## Dataset and target

The project uses the [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients): 30,000 customer records, 23 model features, and the binary target `default payment next month`. The positive class accounts for 22.12% of records.

Citation: Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C55S3H](https://doi.org/10.24432/C55S3H). The dataset is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Place the supplied workbook at `data/default-of-credit-card-clients.xls`. Raw data is intentionally ignored by Git. The loader drops the unique `ID`, maps undocumented education codes `0`, `5`, and `6` to `other` (`4`), and maps undocumented marriage code `0` to `other` (`3`).

## Setup

From this project directory on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m credit_default_risk.train
uvicorn credit_default_risk.api:app --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

See the [experiment report](notebooks/exploratory-analysis.ipynb) for training-partition exploration, model selection, final test diagnostics, and aggregate subgroup results.

## Implementation approach

- A fixed seed and stratification create 60% training, 20% validation, and 20% test partitions.
- Preprocessing is fitted inside scikit-learn pipelines to prevent leakage.
- Numeric features are median-imputed and standardized for logistic regression; categorical fields are most-frequent-imputed and one-hot encoded.
- Calibrated logistic regression is the baseline. Calibrated histogram gradient boosting is the nonlinear tree candidate.
- Five-fold sigmoid calibration is fitted only from training data.
- Model selection ranks candidates by validation ROC-AUC, using lower validation Brier score as the tie-breaker.
- After model selection, the validation set chooses the highest-precision threshold that reaches at least 60% recall.
- The test set is used once for final reporting.

The selected artifact is histogram gradient boosting version `0.1.0`, with an operational threshold of `0.2260`. The risk bands are heuristic: probabilities below half that threshold are `low` risk, values below the operational threshold are `medium`, and values at or above it are `high`.

## Evaluation metrics

Candidate validation metrics at the conventional 0.5 threshold:

| Model | ROC-AUC | Recall | Precision | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.7050 | 0.2193 | 0.6721 | 0.1486 |
| Histogram gradient boosting | 0.7774 | 0.3308 | 0.6582 | 0.1375 |

Selected-model results at threshold `0.2260`:

| Partition | ROC-AUC | Recall | Precision | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.7774 | 0.6006 | 0.4674 | 0.1375 |
| Test | 0.7863 | 0.6368 | 0.4801 | 0.1327 |

Lowering the threshold increases the share of defaults found, at the cost of more non-defaulting customers being included in outreach.

## API behavior and monitoring

`POST /predict` accepts the 23 model features and returns:

```json
{
  "default_probability": 0.6988487349902396,
  "risk_band": "high",
  "model_version": "0.1.0"
}
```

Every successful prediction logs latency, model version, a request ID when supplied, the maximum standardized numeric deviation, the numeric outlier count, and unseen categorical values. These per-request signals are basic operational indicators; production drift decisions require aggregation over a representative window.

## Tests and Docker

```powershell
pytest -q
docker build -t credit-default-risk:local .
docker run --rm -p 8000:8000 credit-default-risk:local
```

Generate `artifacts/model-0.1.0.joblib` before building. The artifact and runtime versions are deliberately aligned because scikit-learn joblib files are version-sensitive.

## Verification results

These thresholds are exercise acceptance criteria and do not establish production suitability.

- Test ROC-AUC: `0.7863` (required: at least `0.75`).
- Test recall: `0.6368` (required: at least `0.60`).
- `/predict` returns a calibrated probability, risk band, and model version.
- Automated tests: `17 passed`, including ten API contract tests.
- Docker: image built and prediction endpoint responded successfully; five post-readiness requests averaged `60.56 ms`, with a maximum of `118.94 ms` (required: below `500 ms`).

## Limitations

- The 2005 Taiwan dataset is a learning benchmark and is not evidence of present-day production performance.
- The release is one cohort without observation timestamps, so the split is stratified rather than temporal.
- Demographic variables can encode or amplify unfairness. Fairness analysis and governance are required before any real use.
- The threshold represents a recall constraint, not measured outreach costs or capacity.
- Drift logs are indicators, not an automated retraining or alerting system.
- Risk bands prioritize outreach only and must not drive adverse credit decisions.
- Predicting default risk does not establish that outreach will change a customer's outcome.

## Future improvements

- Validate on recent, representative, timestamped data using temporal and external holdouts.
- Replace the fixed recall threshold and heuristic risk bands with decisions based on outreach capacity, costs, and expected benefit.
- If intervention data becomes available, model which customers are most likely to benefit from outreach—not only who is most likely to default.
- Evaluate subgroup performance and fairness with appropriate governance.
- Add global and prediction-level explainability for model validation and support review.
- Aggregate drift statistics over time and define alert thresholds.
- Publish a versioned model card if the project develops beyond this learning implementation.
- Benchmark LightGBM, XGBoost, or CatBoost only if they provide measurable value over the current model.
