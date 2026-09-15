import numpy as np
import pytest

from diabetes_readmission_ranking.data import eligible_encounters
from diabetes_readmission_ranking.features import infer_schema, prepare_with_schema
from diabetes_readmission_ranking.inference import rank_discharges
from diabetes_readmission_ranking.modeling import ModelBundle, build_model, fit_calibrators


@pytest.fixture
def bundle_and_frame(encounters):
    frame = eligible_encounters(encounters)
    schema = infer_schema(frame)
    estimator = build_model("catboost", schema)
    estimator.set_params(iterations=10, depth=3)
    features = prepare_with_schema(frame, schema)
    estimator.fit(features, frame["target"])
    raw = estimator.predict_proba(features)[:, 1]
    calibrator = fit_calibrators(frame["target"], raw)["sigmoid"]
    return ModelBundle("catboost", estimator, calibrator, schema, "test-version"), frame


def test_real_catboost_inference_is_ranked(bundle_and_frame):
    bundle, frame = bundle_and_frame
    ranked = rank_discharges(frame.iloc[:10], bundle, capacity=0.2)
    assert ranked["rank"].tolist() == list(range(1, 11))
    assert ranked["selected_for_follow_up"].sum() == 2
    assert np.all(np.diff(ranked["readmission_probability"]) <= 0)


def test_schema_error_is_logged(bundle_and_frame, capsys):
    bundle, frame = bundle_and_frame
    with pytest.raises(ValueError):
        rank_discharges(frame.drop(columns="age"), bundle)
    assert '"event":"schema_error"' in capsys.readouterr().out


def test_delayed_outcome_metrics_are_logged(bundle_and_frame, capsys):
    bundle, frame = bundle_and_frame
    rank_discharges(frame.iloc[:20], bundle)
    assert '"delayed_outcome_metrics"' in capsys.readouterr().out
