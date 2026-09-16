"""Baseline evaluation tests.

References: no external implementation was used in this test file.
"""

import numpy as np
import pandas as pd

from src.baseline_benchmark import evaluate_model, load_data


def test_evaluate_model_smoke() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    metrics = evaluate_model(
        y_true, y_pred, "Smoke Model", fit_time=1.25, prediction_time=0.5
    )

    assert metrics["Model"] == "Smoke Model"
    assert 0.0 <= metrics["Balanced Accuracy"] <= 1.0
    assert 0.0 <= metrics["Macro F1"] <= 1.0
    assert metrics["Fit Time (s)"] == 1.25
    assert metrics["Prediction Time (s)"] == 0.5


def test_load_data_does_not_read_test_split(tmp_path, monkeypatch) -> None:
    sample = pd.DataFrame(
        {
            "alpha": [1.0, 2.0, 3.0],
            "delta": [1.0, 2.0, 3.0],
            "class": ["GALAXY", "QSO", "STAR"],
            "class_encoded": [0, 1, 2],
        }
    )
    sample.to_csv(tmp_path / "train.csv", index=False)
    sample.to_csv(tmp_path / "val.csv", index=False)
    (tmp_path / "test.csv").write_text("reserved\n", encoding="utf-8")

    original_read_csv = pd.read_csv
    read_paths = []

    def recording_read_csv(path, *args, **kwargs):
        read_paths.append(str(path))
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recording_read_csv)

    train, val, split_metadata = load_data(tmp_path)

    assert train.shape == val.shape == sample.shape
    assert set(split_metadata["Split"]) == {"train", "val"}
    assert not any(path.endswith("test.csv") for path in read_paths)
