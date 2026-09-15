import pytest

from diabetes_readmission_ranking.data import eligible_encounters
from diabetes_readmission_ranking.features import infer_schema, input_quality, prepare_with_schema


def test_identifiers_and_outcomes_are_forbidden(encounters):
    schema = infer_schema(eligible_encounters(encounters))
    assert not {"encounter_id", "patient_nbr", "readmitted", "target"} & set(schema.columns)


def test_unseen_categories_are_reported(encounters):
    eligible = eligible_encounters(encounters)
    schema = infer_schema(eligible)
    scoring = eligible.iloc[:1].copy()
    scoring.loc[:, "race"] = "New group"
    assert input_quality(scoring, schema)["unseen_categories"] == {"race": 1}


def test_missing_required_column_is_rejected(encounters):
    eligible = eligible_encounters(encounters)
    schema = infer_schema(eligible)
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_with_schema(eligible.drop(columns="age"), schema)
