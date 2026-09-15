# Diabetes Readmission Ranking

## Goal

Rank eligible diabetes discharges for limited follow-up by estimated 30-day readmission risk. This supports care prioritization; it is not a diagnosis, a treatment recommendation, or a basis for denying care.

## Dataset and target

The project uses the [UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It contains 101,766 encounters from 1999–2008.

The positive target is `readmitted == "<30"`. Encounters ending in death or hospice are ineligible. `encounter_id`, `patient_nbr`, `readmitted`, and the derived target are excluded from features. Patient identifiers are used only to prevent partition overlap.

## Setup instructions

From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,notebook]"
```

Place `diabetic_data.csv` and `IDS_mapping.csv` in `data/`, then train:

```powershell
train-readmission-ranker
```

Rank a batch:

```powershell
rank-discharges --input data/discharges.csv --output artifacts/ranked.csv --log logs/scoring.jsonl
```

## Implementation approach

- Apply explicit discharge eligibility rules and convert `<30` to the binary target.
- Create reproducible 70%/15%/15% train, calibration, and final-test partitions by `patient_nbr`.
- Compare dummy, regularized logistic regression, and CatBoost using patient-grouped three-fold cross-validation on training data.
- Select the highest mean validation PR-AUC; resolve exact ties by model name in ascending order.
- Choose sigmoid or isotonic calibration by calibration-partition Brier score, without consulting the final test.
- Rank final-test and scoring batches at a simulated top-10% follow-up capacity.

CatBoost suits the mixed numeric and categorical data, missing values, and nonlinear interactions. Logistic regression remains an interpretable baseline. Deep learning adds complexity without a clear advantage for this medium-sized tabular dataset; foundation models and agentic systems do not improve the supervised ranking objective.

## Evaluation metrics

- PR-AUC for imbalanced ranking quality
- Brier score for probability accuracy
- Readmissions captured and recall within the top 10%
- Recall and false-negative rate by race, gender, and age, with sample counts

## Verification results

CatBoost was selected without using the final test set.

| Model | Validation PR-AUC | Test PR-AUC | Test readmissions captured at 10% | Test recall at 10% |
|---|---:|---:|---:|---:|
| Dummy | 0.1133 | 0.1161 | 186 | 0.1082 |
| Logistic regression | 0.2116 | 0.2184 | 390 | 0.2269 |
| CatBoost | **0.2381** | **0.2372** | **445** | **0.2589** |
| Random prioritization | — | — | 194 | 0.1129 |

Isotonic calibration reduced final-test Brier score from `0.09666` to `0.09657`. All partition overlaps are zero. The run used `537.74 MB` peak memory, below the 6 GB limit. Full metrics and subgroup evidence are in [verification.json](verification.json), the executed analysis is in [experiment-analysis.ipynb](notebooks/experiment-analysis.ipynb), the sanitized output example is in [ranked-sample.csv](examples/ranked-sample.csv), and intended-use details are in [MODEL_CARD.md](MODEL_CARD.md).

These are exercise results on historical data and do not establish clinical or production suitability.

## Docker

```powershell
docker build -t diabetes-readmission-ranking .
docker run --rm `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/artifacts:/app/artifacts:ro" `
  -v "${PWD}/logs:/app/logs" `
  diabetes-readmission-ranking `
  --input /app/data/discharges.csv `
  --output /app/logs/ranked.csv `
  --model /app/artifacts/readmission_ranker.joblib `
  --log /app/logs/scoring.jsonl
```

## Limitations

The dataset is historical US hospital data from 1999–2008. Its population, coding, care pathways, and outcomes may not represent current settings. Missingness is substantial in several variables. Subgroup estimates are descriptive and unstable for small groups. Risk prediction does not show that follow-up will prevent readmission.

## Future improvements

- Validate on recent, representative data and external health systems.
- Define capacity using operational constraints, intervention costs, and expected benefit.
- Evaluate whether outreach changes outcomes using intervention data.
- Review subgroup performance with clinical, statistical, and governance expertise.
- Monitor delayed outcomes, calibration, missingness, and category drift over time.
