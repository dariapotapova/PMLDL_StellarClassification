"""Verify reported baseline tables and neural checkpoints on validation data.

References
----------
scikit-learn classification metrics:
https://scikit-learn.org/stable/api/sklearn.metrics.html

PyTorch serialization semantics:
https://docs.pytorch.org/docs/stable/notes/serialization.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.model import TabularResNetClassifier
from src.prepare_data import clean_sentinel_values, sha256
from src.reproducibility import configure_determinism


def neural_metrics(root: Path, checkpoint: str, dropout: float, use_layer_norm: bool = False) -> dict[str, float]:
    """Evaluate a saved Tabular ResNet checkpoint on the fixed validation split."""

    train = pd.read_csv(root / "data/train.csv")
    val = pd.read_csv(root / "data/val.csv")
    target_columns = ["class", "class_encoded"]
    X_train = clean_sentinel_values(train.drop(columns=target_columns))
    X_val = clean_sentinel_values(val.drop(columns=target_columns))
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = imputer.fit_transform(X_train)
    scaler.fit(train_values)
    val_values = scaler.transform(imputer.transform(X_val))

    model = TabularResNetClassifier(dropout=dropout, use_layer_norm=use_layer_norm)
    state = torch.load(root / checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state)
    model.eval()
    batches = []
    with torch.no_grad():
        for values in np.array_split(val_values, 60):
            batches.append(model(torch.tensor(values, dtype=torch.float32)).argmax(1))
    predictions = torch.cat(batches)
    expected = val["class_encoded"].to_numpy()
    actual = predictions.numpy()
    return {
        "balanced_accuracy": balanced_accuracy_score(expected, actual),
        "macro_f1": f1_score(expected, actual, average="macro"),
        "accuracy": accuracy_score(expected, actual),
    }


def baseline_metrics(root: Path) -> dict[str, dict[str, float]]:
    """Read stable metrics created by the just-completed baseline run."""

    table = pd.read_csv(root / "reports/tables/baseline_metrics.csv")
    return {
        row["Model"]: {
            "balanced_accuracy": row["Balanced Accuracy"],
            "macro_f1": row["Macro F1"],
            "accuracy": row["Accuracy"],
        }
        for _, row in table.iterrows()
    }


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("reproduction_outputs/verified_metrics.json")
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    configure_determinism()
    reference = json.loads((root / "reports/reference_metrics.json").read_text())
    tolerance = float(reference["absolute_tolerance"])
    experiment_tolerances = reference.get("experiment_tolerances", {})

    for checkpoint, expected_hash in reference["checkpoints"].items():
        actual_hash = sha256(root / checkpoint)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Checkpoint checksum mismatch for {checkpoint}.")

    actual = baseline_metrics(root)
    actual["Tabular ResNet"] = neural_metrics(
        root, "models/tabular_resnet.pt", dropout=0.10
    )
    actual["Tuned Tabular ResNet"] = neural_metrics(
        root, "models/tabular_resnet_tuned.pt", dropout=0.05
    )
    failures = []
    for experiment, expected_metrics in reference["validation_metrics"].items():
        experiment_tolerance = float(experiment_tolerances.get(experiment, tolerance))
        for metric, expected in expected_metrics.items():
            observed = actual[experiment][metric]
            if not np.isclose(
                observed, expected, rtol=0.0, atol=experiment_tolerance
            ):
                failures.append(
                    f"{experiment} {metric}: expected {expected}, observed {observed}"
                )
    if failures:
        raise RuntimeError("Metric verification failed:\n" + "\n".join(failures))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
    print(json.dumps(actual, indent=2, sort_keys=True))
    if experiment_tolerances:
        print(
            f"All metrics match default absolute tolerance {tolerance}; "
            f"documented overrides: {experiment_tolerances}."
        )
    else:
        print(f"All metrics match strict absolute tolerance {tolerance}.")


if __name__ == "__main__":
    main()
