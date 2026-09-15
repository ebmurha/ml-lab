# Model card: diabetes readmission ranker

## Intended use

This learning model ranks eligible diabetes discharges for a simulated follow-up program with capacity for 10% of a batch. It may support technical evaluation of prioritization workflows.

It must not be used for diagnosis, treatment, denial of care, automated clinical decisions, or deployment without current external validation and appropriate clinical governance.

## Model and data

- Model: calibrated CatBoost classifier
- Target: readmission within 30 days
- Data: [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), CC BY 4.0
- Period: 1999–2008
- Eligible encounters: 99,343
- Split: patient-separated 70% training, 15% calibration, 15% final test

## Selection and calibration

Dummy, regularized logistic regression, and CatBoost were compared using mean PR-AUC across three patient-grouped training folds. CatBoost achieved the highest value (`0.2381`). Isotonic calibration was selected using only the calibration partition.

## Final-test performance

- PR-AUC: `0.2372`
- Brier score before calibration: `0.09666`
- Brier score after calibration: `0.09657`
- Readmissions captured in the top 10%: `445` of `1,719`
- Recall at 10%: `0.2589`

At the same capacity, logistic regression captured 390 readmissions and seeded random prioritization captured 194.

## Subgroup evaluation

Recall and false-negative rate are reported with denominators in `verification.json`. Female and male recall were `0.2661` and `0.2497`. Race and age estimates vary more, with very small positive counts in several groups; these differences must not be interpreted as proof of fairness or unfairness.

Race, gender, and age are evaluated because failures may be distributed unevenly. Reporting them does not justify their clinical use, and aggregate categories can hide intersectional harms.

## Limitations and risks

- Historical data may not transfer to current patients or other health systems.
- The dataset does not establish whether follow-up causes better outcomes.
- Missing values, coding conventions, and selection into the dataset can introduce bias.
- Readmission can reflect access, social conditions, and care processes rather than patient need alone.
- A fixed 10% capacity is an exercise assumption, not a clinical policy.
- Small subgroup samples make several estimates unstable.

## Monitoring

Batch scoring logs model version, schema failures, missing and unseen categories, latency, score distribution, and delayed PR-AUC/Brier metrics when outcomes are supplied. Logs must be handled under the deploying organization's privacy and security controls.
