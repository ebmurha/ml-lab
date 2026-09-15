import json
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _report():
    return json.loads((PROJECT_ROOT / "verification.json").read_text(encoding="utf-8"))


def test_objective_checks_pass():
    report = _report()
    assert not any(report["dataset"]["patient_overlap"].values())
    assert report["model"]["forbidden_feature_overlap"] == []
    assert report["selection"]["final_test_used_for_selection"] is False
    assert report["calibration"]["test_brier_after"] <= report["calibration"]["test_brier_before"]
    assert report["verification"]["peak_memory_mb"] < 6 * 1024


def test_selected_model_beats_comparators_at_capacity():
    test = _report()["test"]
    selected = test["catboost"]["readmissions_captured"]
    assert selected > test["logistic_regression"]["readmissions_captured"]
    assert selected > test["random_prioritization"]["readmissions_captured"]


def test_notebook_is_executed_and_sanitized():
    path = PROJECT_ROOT / "notebooks" / "experiment-analysis.ipynb"
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells and all(cell.execution_count is not None for cell in code_cells)
    assert not any(
        output.output_type == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    output_text = json.dumps(
        [cell.get("outputs", []) for cell in code_cells], default=str
    ).lower()
    forbidden = ("c:\\\\users", "/users/", "/home/", "appdata", "site-packages", "traceback")
    assert not any(value in output_text for value in forbidden)
    assert path.stat().st_size < 1_000_000
