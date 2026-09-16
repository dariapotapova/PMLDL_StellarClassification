"""Tests for deterministic data preparation and canonical configuration.

References: no external implementation was used in this test file.
"""

import json
from pathlib import Path

import pandas as pd
import torch

from src.prepare_data import LABEL_MAPPING, split_dataframe
from src.reproducibility import PROJECT_SEED, configure_determinism


def test_project_seed_and_torch_generator_are_fixed() -> None:
    assert PROJECT_SEED == 42
    first = torch.rand(4, generator=configure_determinism())
    second = torch.rand(4, generator=configure_determinism())
    assert torch.equal(first, second)


def test_split_is_deterministic_and_stratified() -> None:
    rows = []
    for class_name, encoded in LABEL_MAPPING.items():
        for index in range(20):
            rows.append(
                {
                    "alpha": float(index),
                    "class": class_name,
                    "class_encoded": encoded,
                }
            )
    frame = pd.DataFrame(rows)
    first = split_dataframe(frame)
    second = split_dataframe(frame)
    assert {name: len(value) for name, value in first.items()} == {
        "train": 42,
        "val": 9,
        "test": 9,
    }
    for name in first:
        assert first[name].equals(second[name])
        assert set(first[name]["class_encoded"]) == {0, 1, 2}


def test_canonical_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        "requirements.lock.txt",
        "scripts/reproduce.sh",
        "src/prepare_data.py",
        "src/reproducibility.py",
        "src/verify_results.py",
        "src/verify_environment.py",
        "reports/reference_metrics.json",
        "reports/reproducibility.md",
    ]
    assert all((root / path).is_file() for path in required)


def test_dependency_files_use_exact_versions() -> None:
    root = Path(__file__).resolve().parents[1]
    for filename in ("requirements.txt", "requirements.lock.txt"):
        lines = [
            line.strip()
            for line in (root / filename).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert lines
        assert all(line.count("==") == 1 for line in lines)


def test_reference_metrics_cover_required_experiments() -> None:
    root = Path(__file__).resolve().parents[1]
    reference = json.loads(
        (root / "reports/reference_metrics.json").read_text(encoding="utf-8")
    )
    assert set(reference["validation_metrics"]) == {
        "Logistic Regression",
        "Random Forest",
        "CatBoost",
        "Tabular ResNet",
        "Tuned Tabular ResNet",
    }
    for metrics in reference["validation_metrics"].values():
        assert set(metrics) == {"balanced_accuracy", "macro_f1", "accuracy"}
        assert all(0.0 <= value <= 1.0 for value in metrics.values())
