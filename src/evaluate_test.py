"""One-time final evaluation of all models on the reserved test split.

Protocol:
- test.csv is loaded only here, exactly once, after all model selection;
- baselines are rebuilt via build_models() imported from
  src.baseline_benchmark, so configurations and seeds are identical
  to the validation benchmark;
- neural checkpoints are the ones saved by src/train_model.py;
- a marker file (reports/.test_evaluated) blocks accidental re-runs;
  pass --force to redo the whole final evaluation deliberately.

Run from the repository root:

    python -m src.evaluate_test
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.baseline_benchmark import (
    CLASS_ORDER,
    LABEL_ORDER,
    SUMMARY_COLUMNS,
    TARGET_CLASS,
    TARGET_ENCODED,
    _package_versions,
    _sorted_summary,
    _validate_label_mapping,
    build_models,
    calculate_per_class_metrics,
    evaluate_model,
    fit_and_predict_models,
    load_data,
    prepare_features,
    resolve_paths,
)
from src.model import TabularResNetClassifier
from src.prepare_data import clean_sentinel_values

RESNET_RUNS = {
    "Tabular ResNet (initial/LN baseline)": (
        "tabular_resnet.pt", "proposed", {"use_layer_norm": True, "dropout": 0.10},
    ),
    "Tabular ResNet (ablation/BN control)": (
        "tabular_resnet_ablation_norm.pt", "ablation", {"use_layer_norm": False, "dropout": 0.10},
    ),
    "Tabular ResNet (tuned/LN improved)": (
        "tabular_resnet_tuned.pt", "improved", {"use_layer_norm": True, "dropout": 0.05},
    ),
}


def resnet_predict(
    checkpoint: Path, X_scaled: np.ndarray, **model_kwargs: object
) -> tuple[np.ndarray, np.ndarray]:
    """Load a saved Tabular ResNet checkpoint and predict class labels/probabilities.

    All keyword arguments are forwarded to ``TabularResNetClassifier`` so that
    LayerNorm / BatchNorm variants, different dropout values, or other
    constructor options can be matched to the checkpoint that was saved with
    those settings.
    """
    model = TabularResNetClassifier(**model_kwargs)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, dtype=torch.float32))
        proba = torch.softmax(logits, dim=1).numpy()
    return proba.argmax(axis=1), proba


def plot_metric_comparison(summary: pd.DataFrame, output: Path) -> None:
    melted = summary.melt(
        id_vars="Model",
        value_vars=["Balanced Accuracy", "Macro F1", "Accuracy"],
        var_name="Metric",
        value_name="Score",
    )
    palette = dict(zip(summary["Model"], sns.color_palette("Set2", len(summary))))
    plt.figure(figsize=(10.5, 6.0))
    ax = sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette=palette)
    ax.set_title("Overall Test Metric Comparison")
    ax.set_ylim(0, 1.08)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    ax.legend(title="Model", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_per_class_f1(per_class: pd.DataFrame, models: list[str], output: Path) -> None:
    palette = dict(zip(models, sns.color_palette("Set2", len(models))))
    plt.figure(figsize=(9.5, 5.5))
    ax = sns.barplot(
        data=per_class, x="Class", y="F1", hue="Model",
        order=CLASS_ORDER, hue_order=models, palette=palette,
    )
    ax.set_title("Per-class Test F1 Comparison")
    ax.set_ylim(0, 1.02)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_confusions(y_true, predictions: dict[str, np.ndarray], output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5))
    for ax, name in zip(axes.ravel(), list(predictions)):
        normalized = confusion_matrix(
            y_true, predictions[name], labels=LABEL_ORDER, normalize="true"
        )
        sns.heatmap(
            normalized,
            annot=np.vectorize(lambda v: f"{v:.1%}")(normalized),
            fmt="", cmap="Greens", vmin=0, vmax=1, cbar=False,
            xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Test Confusion Matrices: Row-normalized", fontsize=14)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--force", action="store_true",
                        help="re-run even if the test marker file already exists")
    args = parser.parse_args()

    paths = resolve_paths(args.repo_root)
    marker = paths.reports_dir / ".test_evaluated"
    if marker.exists() and not args.force:
        raise SystemExit(
            "The test set has already been evaluated (single-evaluation protocol). "
            "Pass --force only if you deliberately want to redo the final evaluation."
        )

    # Train data only: fitted models and preprocessing. val.csv is loaded by
    # prepare_features for validation checks; test.csv is read exactly once below.
    train, val, _ = load_data(paths.data_dir)
    X_train, y_train, _, _, feature_columns, _, _ = prepare_features(train, val)
    X_train = clean_sentinel_values(X_train)

    test_path = paths.data_dir / "test.csv"
    test = pd.read_csv(test_path)  # <-- the single load of the test split
    _validate_label_mapping(test, "test")
    needed = [TARGET_CLASS, TARGET_ENCODED, *feature_columns]
    missing = [c for c in needed if c not in test.columns]
    if missing:
        raise ValueError(f"test.csv is missing required columns: {missing}")
    X_test = test[feature_columns].copy()
    X_test = clean_sentinel_values(X_test)
    y_test = test[TARGET_ENCODED].copy()
    print(f"Test split loaded once: {len(y_test)} rows.")

    # --- Baselines: same builders, same fit/predict code as the validation benchmark.
    models = build_models()
    baseline_results = fit_and_predict_models(models, X_train, y_train, X_test)

    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    model_type: dict[str, str] = {}
    for result in baseline_results:
        row = evaluate_model(y_test, result.y_pred, result.name,
                             result.fit_time, result.prediction_time)
        proba = result.estimator.predict_proba(X_test)
        row["ROC-AUC (OvR macro)"] = roc_auc_score(
            y_test, proba, multi_class="ovr", average="macro", labels=LABEL_ORDER)
        rows.append(row)
        predictions[result.name] = result.y_pred
        model_type[result.name] = "baseline"

    # --- Neural models: preprocessing fitted on train only, then saved checkpoints.
    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaler = StandardScaler().fit(imputer.transform(X_train))
    X_test_scaled = scaler.transform(imputer.transform(X_test)).astype(np.float32)

    per_class_frames = [
        calculate_per_class_metrics(y_test, r.y_pred, r.name) for r in baseline_results
    ]
    for display_name, (filename, kind, model_kwargs) in RESNET_RUNS.items():
        checkpoint = paths.repo_root / "models" / filename
        if not checkpoint.exists():
            print(f"[warn] Skipping {display_name}: checkpoint not found at {checkpoint}")
            continue
        y_pred, proba = resnet_predict(checkpoint, X_test_scaled, **model_kwargs)
        row = evaluate_model(y_test, y_pred, display_name)
        row["ROC-AUC (OvR macro)"] = roc_auc_score(
            y_test, proba, multi_class="ovr", average="macro", labels=LABEL_ORDER)
        rows.append(row)
        predictions[display_name] = y_pred
        model_type[display_name] = kind
        per_class_frames.append(calculate_per_class_metrics(y_test, y_pred, display_name))

    # --- Tables.
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(rows)
    summary["Type"] = summary["Model"].map(model_type)
    column_order = (["Model", "Type"]
                    + [c for c in SUMMARY_COLUMNS if c != "Model"]
                    + ["ROC-AUC (OvR macro)"])
    summary = summary[column_order]
    summary = _sorted_summary(summary)
    summary.to_csv(paths.tables_dir / "test_metrics.csv", index=False)

    per_class = pd.concat(per_class_frames, ignore_index=True)
    per_class.to_csv(paths.tables_dir / "test_per_class_metrics.csv", index=False)

    # --- Figures.
    plot_metric_comparison(summary, paths.figures_dir / "test_metrics_comparison.png")
    plot_per_class_f1(per_class, list(summary["Model"]),
                      paths.figures_dir / "test_per_class_f1.png")
    plot_confusions(y_test, predictions,
                    paths.figures_dir / "test_confusion_matrices_overview.png")

    # --- Frozen results record + single-run marker.
    payload = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_test": int(len(y_test)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        **{f"package_{k.lower()}": v for k, v in _package_versions().items()
           if k != "Python"},
        "metrics": rows,
    }
    (paths.reports_dir / "test_results.json").write_text(json.dumps(payload, indent=2))
    marker.write_text(datetime.now(timezone.utc).isoformat())

    print("\n=== FINAL TEST RESULTS (single evaluation) ===")
    print(summary.drop(columns=["Type"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nSaved:")
    print(" - reports/tables/test_metrics.csv")
    print(" - reports/tables/test_per_class_metrics.csv")
    print(" - reports/figures/test_metrics_comparison.png")
    print(" - reports/figures/test_per_class_f1.png")
    print(" - reports/figures/test_confusion_matrices_overview.png")
    print(" - reports/test_results.json")
    print(f" - {marker.name} (re-runs are now blocked; --force overrides)")


if __name__ == "__main__":
    main()       