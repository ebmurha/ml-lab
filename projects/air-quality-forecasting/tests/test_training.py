from air_quality_forecasting.train import build_model_version


def test_model_version_changes_with_data_configuration_or_source(monkeypatch) -> None:
    monkeypatch.setattr(
        "air_quality_forecasting.train.source_digest", lambda: "c" * 64
    )
    first, first_config, first_code = build_model_version(
        "model", "a" * 64, ["feature_a"]
    )
    changed_data, _, _ = build_model_version("model", "b" * 64, ["feature_a"])
    changed_config, _, _ = build_model_version("model", "a" * 64, ["feature_b"])
    monkeypatch.setattr(
        "air_quality_forecasting.train.source_digest", lambda: "d" * 64
    )
    changed_code, _, _ = build_model_version("model", "a" * 64, ["feature_a"])

    assert len({first, changed_data, changed_config, changed_code}) == 4
    assert len(first_config) == 64
    assert first_code == "c" * 64
