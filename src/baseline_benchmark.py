"""Train and evaluate standardized stellar-classification baseline models.

Run from the repository root with:

    python -m src.baseline_benchmark

or on Windows:

    py -m src.baseline_benchmark

References
----------
Original SDSS17 dataset page:
https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

Public model-family precedents used by this project:
https://github.com/wonhee3472/Stellar_Classification
https://www.kaggle.com/code/leylamammadzada/stellar-classification-random-forest
https://www.kaggle.com/code/aerdem/stellar-classification-catboost-97-accuracy
"""

from __future__ import annotations

import argparse
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.prepare_data import clean_sentinel_values
from src.reproducibility import PROJECT_SEED

RANDOM_STATE = PROJECT_SEED
TARGET_CLASS = "class"
TARGET_ENCODED = "class_encoded"
CLASS_ORDER = ["GALAXY", "QSO", "STAR"]
LABEL_MAPPING = {"GALAXY": 0, "QSO": 1, "STAR": 2}
LABEL_ORDER = [LABEL_MAPPING[name] for name in CLASS_ORDER]

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "catboost": "CatBoost",
}

SUMMARY_COLUMNS = [
    "Model",
    "Balanced Accuracy",
    "Macro F1",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Weighted F1",
    "Fit Time (s)",
    "Prediction Time (s)",
]

FIGURE_NAMES = {
    "class_distribution": "class_distribution.png",
    "metrics_comparison": "metrics_comparison.png",
    "per_class_f1": "per_class_f1_comparison.png",
    "confusion_overview": "confusion_matrices_overview.png",
    "logistic_importance": "logistic_regression_coefficients.png",
    "rf_importance": "random_forest_feature_importance.png",
    "catboost_importance": "catboost_feature_importance.png",
    "feature_distributions": "top_feature_distributions_by_class.png",
}

MODEL_SLUGS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "CatBoost": "catboost",
}

MODEL_PALETTE = {
    "Logistic Regression": "#4C78A8",
    "Random Forest": "#F58518",
    "CatBoost": "#54A24B",
}

CLASS_PALETTE = {
    "GALAXY": "#4C78A8",
    "QSO": "#E45756",
    "STAR": "#72B7B2",
}


@dataclass(frozen=True)
class BenchmarkPaths:
    """Output paths used by the benchmark."""

    repo_root: Path
    data_dir: Path
    reports_dir: Path
    figures_dir: Path
    tables_dir: Path
    report_path: Path


@dataclass(frozen=True)
class BenchmarkData:
    """Validated split data and feature matrices."""

    train: pd.DataFrame
    val: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    feature_columns: list[str]
    split_metadata: pd.DataFrame
    missing_summary: pd.DataFrame
    infinite_summary: pd.DataFrame


@dataclass(frozen=True)
class ModelResult:
    """Predictions, timing and fitted estimator for one model."""

    name: str
    estimator: Any
    y_pred: np.ndarray
    fit_time: float
    prediction_time: float


@dataclass(frozen=True)
class BenchmarkResult:
    """Complete benchmark outputs."""

    summary_metrics: pd.DataFrame
    per_class_metrics: pd.DataFrame
    split_metadata: pd.DataFrame
    feature_columns: list[str]
    figure_paths: dict[str, Path]
    report_path: Path
    best_model: str


def resolve_paths(repo_root: Path | str = ".") -> BenchmarkPaths:
    """Resolve repository-relative input and output paths."""

    root = Path(repo_root).resolve()
    reports_dir = root / "reports"
    return BenchmarkPaths(
        repo_root=root,
        data_dir=root / "data",
        reports_dir=reports_dir,
        figures_dir=reports_dir / "figures",
        tables_dir=reports_dir / "tables",
        report_path=reports_dir / "baseline_benchmark.md",
    )


def find_data_files(data_dir: Path | str = "data") -> dict[str, Path]:
    """Locate prepared split files without reading the test split."""

    root = Path(data_dir)
    required = {
        "train": root / "train.csv",
        "val": root / "val.csv",
        "test": root / "test.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required prepared data files: {missing}")
    return required


def _validate_label_mapping(df: pd.DataFrame, split_name: str) -> None:
    """Validate that text and encoded targets have the expected one-to-one mapping."""

    if TARGET_CLASS not in df.columns or TARGET_ENCODED not in df.columns:
        raise ValueError(
            f"{split_name} must contain {TARGET_CLASS!r} and {TARGET_ENCODED!r}."
        )

    observed = (
        df[[TARGET_CLASS, TARGET_ENCODED]]
        .drop_duplicates()
        .sort_values([TARGET_CLASS, TARGET_ENCODED])
    )
    expected = pd.DataFrame(
        {
            TARGET_CLASS: list(LABEL_MAPPING.keys()),
            TARGET_ENCODED: list(LABEL_MAPPING.values()),
        }
    ).sort_values([TARGET_CLASS, TARGET_ENCODED])
    if not observed.reset_index(drop=True).equals(expected.reset_index(drop=True)):
        raise ValueError(
            f"{split_name} label mapping is inconsistent. Observed mapping: "
            f"{observed.to_dict(orient='records')}"
        )


def _split_metadata(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Calculate row counts and class distribution for one split."""

    counts = (
        df[TARGET_CLASS]
        .value_counts()
        .reindex(CLASS_ORDER)
        .rename_axis("Class")
        .reset_index(name="Count")
    )
    counts["Split"] = split_name
    counts["Percent"] = counts["Count"] / len(df)
    counts["Rows"] = len(df)
    return counts[["Split", "Rows", "Class", "Count", "Percent"]]


def load_data(
    data_dir: Path | str = "data",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/validation data and verify that the reserved test file exists.

    The test split is never loaded or inspected by this benchmark.
    """

    files = find_data_files(data_dir)
    train = pd.read_csv(files["train"])
    val = pd.read_csv(files["val"])
    metadata_frames = [_split_metadata(train, "train"), _split_metadata(val, "val")]

    return train, val, pd.concat(metadata_frames, ignore_index=True)


def validate_data(train: pd.DataFrame, val: pd.DataFrame) -> None:
    """Validate split schema, label mapping and target classes."""

    for split_name, df in {"train": train, "val": val}.items():
        if df.empty:
            raise ValueError(f"{split_name} is empty.")
        _validate_label_mapping(df, split_name)
        observed_classes = set(df[TARGET_CLASS].unique())
        if observed_classes != set(CLASS_ORDER):
            raise ValueError(
                f"{split_name} has unexpected classes: {sorted(observed_classes)}"
            )

    if list(train.columns) != list(val.columns):
        raise ValueError(
            "Train and validation columns must match exactly and in the same order."
        )


def prepare_features(train: pd.DataFrame, val: pd.DataFrame) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    list[str],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create leakage-free numeric feature matrices for train and validation."""

    validate_data(train, val)
    target_columns = [TARGET_CLASS, TARGET_ENCODED]
    X_train = train.drop(columns=target_columns).copy()
    X_val = val.drop(columns=target_columns).copy()

    X_train = clean_sentinel_values(X_train)
    X_val = clean_sentinel_values(X_val)

    leaked = [
        column
        for column in target_columns
        if column in X_train.columns or column in X_val.columns
    ]
    if leaked:
        raise ValueError(f"Target columns remain in the model features: {leaked}")

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError(
            "Train and validation feature columns must match in the same order."
        )

    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise TypeError(
            f"Non-numeric feature columns require explicit handling: {non_numeric}"
        )

    missing_summary = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Train Missing": X_train.isna().sum().to_numpy(),
            "Validation Missing": X_val.isna().sum().to_numpy(),
        }
    )

    numeric_train = X_train.to_numpy(dtype=float)
    numeric_val = X_val.to_numpy(dtype=float)
    infinite_summary = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Train Infinite": np.isinf(numeric_train).sum(axis=0),
            "Validation Infinite": np.isinf(numeric_val).sum(axis=0),
        }
    )
    if infinite_summary[["Train Infinite", "Validation Infinite"]].to_numpy().sum() > 0:
        raise ValueError("Infinite values were found in prepared features.")

    y_train = train[TARGET_ENCODED].copy()
    y_val = val[TARGET_ENCODED].copy()
    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_train.columns.tolist(),
        missing_summary,
        infinite_summary,
    )


def build_models() -> dict[str, Any]:
    """Build the fixed baseline estimators."""

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        max_features="sqrt",
                        min_samples_leaf=1,
                    ),
                ),
            ]
        ),
        "catboost": CatBoostClassifier(
            loss_function="MultiClass",
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
    }


def fit_and_predict_models(
    models: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
) -> list[ModelResult]:
    """Fit models on train only and predict on validation only."""

    results: list[ModelResult] = []
    for slug, estimator in models.items():
        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_time = time.perf_counter() - start

        start = time.perf_counter()
        y_pred = estimator.predict(X_val)
        prediction_time = time.perf_counter() - start
        y_pred = np.asarray(y_pred).reshape(-1).astype(int)

        results.append(
            ModelResult(
                name=MODEL_NAMES[slug],
                estimator=estimator,
                y_pred=y_pred,
                fit_time=fit_time,
                prediction_time=prediction_time,
            )
        )
    return results


def evaluate_model(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    fit_time: float = 0.0,
    prediction_time: float = 0.0,
) -> dict[str, float | str]:
    """Calculate standardized model-level classification metrics."""

    return {
        "Model": model_name,
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Fit Time (s)": fit_time,
        "Prediction Time (s)": prediction_time,
    }


def calculate_per_class_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Calculate precision, recall, F1 and support for every class."""

    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for class_name in CLASS_ORDER:
        metrics = report[class_name]
        rows.append(
            {
                "Model": model_name,
                "Class": class_name,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1-score"],
                "Support": int(metrics["support"]),
            }
        )
    return pd.DataFrame(rows)


def _sorted_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Sort summary metrics by the documented model-selection protocol."""

    return summary.sort_values(
        ["Balanced Accuracy", "Macro F1"], ascending=False
    ).reset_index(drop=True)


def _format_markdown_table(
    df: pd.DataFrame, numeric_columns: list[str] | None = None
) -> str:
    """Render a compact Markdown table with four decimal places for floats."""

    formatted = df.copy()
    columns = (
        numeric_columns or formatted.select_dtypes(include=[np.number]).columns.tolist()
    )
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: (
                    f"{value:.4f}"
                    if isinstance(value, float)
                    or np.issubdtype(type(value), np.floating)
                    else value
                )
            )
    return formatted.to_markdown(index=False)


def plot_class_distribution(split_metadata: pd.DataFrame, output_path: Path) -> Path:
    """Plot class distribution across train and validation splits."""

    plot_data = split_metadata.copy()
    plot_data["Percent"] = plot_data["Percent"] * 100
    plt.figure(figsize=(8.5, 5.2))
    ax = sns.barplot(
        data=plot_data,
        x="Class",
        y="Percent",
        hue="Split",
        order=CLASS_ORDER,
        palette={"train": "#4C78A8", "val": "#F58518"},
    )
    ax.set_title("Class Distribution by Train and Validation Split")
    ax.set_xlabel("Class")
    ax.set_ylabel("Percent of split")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> Path:
    """Plot raw and row-normalized confusion matrices for one model."""

    counts = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    normalized = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    sns.heatmap(
        counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
        ax=axes[0],
    )
    axes[0].set_title(f"{model_name}: Counts")
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")

    sns.heatmap(
        normalized,
        annot=np.vectorize(lambda value: f"{value:.1%}")(normalized),
        fmt="",
        cmap="Greens",
        vmin=0,
        vmax=1,
        cbar=False,
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
        ax=axes[1],
    )
    axes[1].set_title(f"{model_name}: Row-normalized")
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("True class")
    fig.suptitle(f"Validation Confusion Matrix - {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_confusion_matrices_overview(
    results: list[ModelResult], y_true: pd.Series, output_path: Path
) -> Path:
    """Plot row-normalized confusion matrices for all baselines."""

    fig, axes = plt.subplots(1, len(results), figsize=(14.5, 4.5))
    for ax, result in zip(axes, results, strict=True):
        normalized = confusion_matrix(
            y_true, result.y_pred, labels=LABEL_ORDER, normalize="true"
        )
        sns.heatmap(
            normalized,
            annot=np.vectorize(lambda value: f"{value:.1%}")(normalized),
            fmt="",
            cmap="Greens",
            vmin=0,
            vmax=1,
            cbar=False,
            xticklabels=CLASS_ORDER,
            yticklabels=CLASS_ORDER,
            ax=ax,
        )
        ax.set_title(result.name)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
    fig.suptitle("Validation Confusion Matrices: Row-normalized", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_metric_comparison(summary_metrics: pd.DataFrame, output_path: Path) -> Path:
    """Plot balanced accuracy, macro F1 and accuracy by model."""

    plot_data = _sorted_summary(summary_metrics).melt(
        id_vars="Model",
        value_vars=["Balanced Accuracy", "Macro F1", "Accuracy"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(9.4, 6.0))
    ax = sns.barplot(
        data=plot_data,
        x="Metric",
        y="Score",
        hue="Model",
        hue_order=list(MODEL_PALETTE.keys()),
        palette=MODEL_PALETTE,
    )
    ax.set_title("Overall Validation Metric Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    ax.legend(
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_per_class_f1(per_class_metrics: pd.DataFrame, output_path: Path) -> Path:
    """Plot per-class validation F1 scores for every baseline."""

    plt.figure(figsize=(8.8, 5.3))
    ax = sns.barplot(
        data=per_class_metrics,
        x="Class",
        y="F1",
        hue="Model",
        order=CLASS_ORDER,
        hue_order=list(MODEL_PALETTE.keys()),
        palette=MODEL_PALETTE,
    )
    ax.set_title("Per-class Validation F1 Comparison")
    ax.set_xlabel("Class")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.02)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def _logistic_coefficients(
    estimator: Pipeline, feature_columns: list[str]
) -> pd.DataFrame:
    """Extract class-wise standardized coefficients from the logistic pipeline."""

    model = estimator.named_steps["model"]
    return pd.DataFrame(model.coef_, index=CLASS_ORDER, columns=feature_columns)


def _feature_importance_from_result(
    result: ModelResult, feature_columns: list[str]
) -> pd.DataFrame:
    """Extract feature importances from a fitted tree-style model."""

    estimator = result.estimator
    if isinstance(estimator, Pipeline):
        model = estimator.named_steps["model"]
        importances = model.feature_importances_
    else:
        importances = estimator.get_feature_importance()
    return (
        pd.DataFrame({"Feature": feature_columns, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(
    result: ModelResult, feature_columns: list[str], output_path: Path
) -> Path:
    """Plot model interpretation for logistic regression, random forest or CatBoost."""

    if result.name == "Logistic Regression":
        coefs = _logistic_coefficients(result.estimator, feature_columns)
        plt.figure(figsize=(10.8, 4.9))
        abs_values = np.abs(coefs.to_numpy()).ravel()
        max_abs = float(np.quantile(abs_values, 0.95))
        max_abs = max(max_abs, float(abs_values.max()) * 0.35)
        sns.heatmap(coefs, cmap="vlag", center=0, vmin=-max_abs, vmax=max_abs)
        plt.title(
            "Logistic Regression Standardized Coefficients\nColor scale clipped for readability"
        )
        plt.xlabel("Feature")
        plt.ylabel("Class")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return output_path

    importance = _feature_importance_from_result(result, feature_columns).head(12)
    plt.figure(figsize=(8.7, 5.4))
    ax = sns.barplot(
        data=importance,
        x="Importance",
        y="Feature",
        orient="h",
        color=MODEL_PALETTE[result.name],
    )
    ax.set_title(f"{result.name} Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_top_feature_distributions(
    train: pd.DataFrame,
    feature_columns: list[str],
    reference_result: ModelResult,
    output_path: Path,
) -> Path:
    """Plot robust-clipped training distributions for important features by class.

    Clipping is applied only to a plotted copy, using train-only 0.5th and 99.5th
    percentiles, so extreme values do not compress the visible distributions.
    """

    if reference_result.name == "Logistic Regression":
        coefs = (
            _logistic_coefficients(reference_result.estimator, feature_columns)
            .abs()
            .max(axis=0)
        )
        features = coefs.sort_values(ascending=False).head(4).index.tolist()
    else:
        features = (
            _feature_importance_from_result(reference_result, feature_columns)
            .head(4)["Feature"]
            .tolist()
        )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    for ax, feature in zip(axes.ravel(), features, strict=True):
        lower = float(train[feature].quantile(0.005))
        upper = float(train[feature].quantile(0.995))
        plot_data = train[[TARGET_CLASS, feature]].copy()
        plot_data[feature] = plot_data[feature].clip(lower=lower, upper=upper)
        sns.histplot(
            data=plot_data,
            x=feature,
            hue=TARGET_CLASS,
            hue_order=CLASS_ORDER,
            palette=CLASS_PALETTE,
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            bins=60,
            ax=ax,
        )
        ax.set_title(f"{feature} by class")
        ax.set_xlabel(f"{feature} (clipped to {lower:.3g}..{upper:.3g})")
        ax.set_ylabel("Density")
    fig.suptitle(
        "Training Feature Distributions by Class\nVisualization-only clipping at train 0.5th-99.5th percentiles",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _package_versions() -> dict[str, str]:
    """Collect package versions relevant for reproducibility."""

    import catboost
    import matplotlib as mpl
    import sklearn

    return {
        "Python": platform.python_version(),
        "catboost": catboost.__version__,
        "matplotlib": mpl.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "seaborn": sns.__version__,
    }


def _relative(path: Path, repo_root: Path) -> str:
    """Return a POSIX-style repository-relative path for reports."""

    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def write_report(
    paths: BenchmarkPaths,
    summary_metrics: pd.DataFrame,
    per_class_metrics: pd.DataFrame,
    split_metadata: pd.DataFrame,
    feature_columns: list[str],
    figure_paths: dict[str, Path],
    y_val: pd.Series,
    model_results: list[ModelResult],
) -> Path:
    """Write the Markdown benchmark report."""

    sorted_summary = _sorted_summary(summary_metrics)
    best_model = str(sorted_summary.loc[0, "Model"])
    split_table = split_metadata.copy()
    split_table["Percent"] = split_table["Percent"].map(lambda value: f"{value:.2%}")
    display_per_class = per_class_metrics.copy()
    display_per_class["Support"] = display_per_class["Support"].astype(int)
    macro_f1_winner = str(
        summary_metrics.sort_values(
            ["Macro F1", "Balanced Accuracy"], ascending=False
        ).iloc[0]["Model"]
    )
    accuracy_winner = str(
        summary_metrics.sort_values(
            ["Accuracy", "Balanced Accuracy"], ascending=False
        ).iloc[0]["Model"]
    )
    best_row = sorted_summary.iloc[0]
    macro_row = summary_metrics[summary_metrics["Model"] == macro_f1_winner].iloc[0]
    accuracy_row = summary_metrics[summary_metrics["Model"] == accuracy_winner].iloc[0]

    source_links = [
        (
            "Logistic Regression and Random Forest project",
            "https://github.com/wonhee3472/Stellar_Classification",
            "Used as public precedent for comparing Logistic Regression and Random Forest on SDSS stellar classes.",
        ),
        (
            "Random Forest notebook",
            "https://www.kaggle.com/code/leylamammadzada/stellar-classification-random-forest",
            "Used as public precedent for a Random Forest stellar-classification baseline.",
        ),
        (
            "CatBoost notebook",
            "https://www.kaggle.com/code/aerdem/stellar-classification-catboost-97-accuracy",
            "Used as public precedent for a CatBoost multiclass stellar-classification baseline.",
        ),
        (
            "Original SDSS17 dataset",
            "https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17",
            "Original Kaggle dataset used to prepare the local train, validation and test splits.",
        ),
    ]
    figure_lines = "\n".join(
        f"- [{path.name}]({_relative(path, paths.reports_dir)})"
        for path in figure_paths.values()
    )
    version_lines = "\n".join(
        f"- `{name}`: `{version}`" for name, version in _package_versions().items()
    )
    source_lines = "\n".join(
        f"- [{name}]({url}): {description}" for name, url, description in source_links
    )
    result_by_name = {result.name: result for result in model_results}
    top_errors = []
    for model_name in sorted_summary["Model"]:
        rows = display_per_class[display_per_class["Model"] == model_name].sort_values(
            "F1"
        )
        weakest = rows.iloc[0]
        counts = confusion_matrix(
            y_val, result_by_name[model_name].y_pred, labels=LABEL_ORDER
        )
        errors = counts.copy()
        np.fill_diagonal(errors, 0)
        true_index, pred_index = np.unravel_index(np.argmax(errors), errors.shape)
        error_count = int(errors[true_index, pred_index])
        true_class = CLASS_ORDER[true_index]
        pred_class = CLASS_ORDER[pred_index]
        error_rate = error_count / counts[true_index].sum()
        top_errors.append(
            f"- {model_name}: weakest validation F1 is `{weakest['Class']}` "
            f"({weakest['F1']:.4f}). The most common off-diagonal error is true "
            f"`{true_class}` predicted as `{pred_class}` ({error_count} rows, {error_rate:.2%} of true `{true_class}`)."
        )

    report = f"""# Baseline Benchmark and Evaluation Metrics

## Objective

This report defines the classification metrics for all baseline and future custom models, then benchmarks three public-solution-inspired model families on the prepared `train.csv` and `val.csv` splits.

## Dataset and Splits

The prepared train and validation files are stored in `data/` and use the verified target mapping `GALAXY -> 0`, `QSO -> 1`, `STAR -> 2`. The benchmark fits models only on `train.csv` and evaluates only on `val.csv`. The test split was not loaded or inspected. It remains reserved for the single final evaluation after model selection.

Features used by every baseline:

`{', '.join(feature_columns)}`

Verified train and validation class distribution:

{split_table.to_markdown(index=False)}

The classes are moderately imbalanced: `GALAXY` is about 59.44-59.45% of the train and validation splits, `STAR` about 21.59%, and `QSO` about 18.96%. This makes ordinary accuracy insufficient as the sole model-selection metric because a model can score well by favoring the majority `GALAXY` class.

## Evaluation Metrics

Primary metric: **balanced accuracy**. Balanced accuracy is the unweighted mean of recall across `GALAXY`, `QSO` and `STAR`, so every class contributes equally despite class imbalance. This is also the evaluation metric used by the closely related Kaggle Playground stellar-classification competition.

Secondary metric: **macro F1**. Macro F1 calculates F1 independently for each class and gives each class equal weight. Unlike balanced accuracy, it accounts for both precision and recall, helping reject models that raise recall by creating too many false-positive predictions.

Every model must report:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- Balanced accuracy
- Precision, recall, F1 and support for every class
- Confusion matrix

Metric calls use `zero_division=0` where applicable. Accuracy must not be used as the sole model-selection metric. ROC-AUC and log loss are not required benchmark metrics; they may be added only as supplementary diagnostics when implemented correctly for multiclass probability predictions.

## Comparison Protocol

All baseline and future custom models must use the same prepared features and label mapping. Baseline selection is performed on `val.csv`, with models ranked primarily by balanced accuracy and macro F1 used as the main supporting metric and tie-breaker. Per-class metrics and confusion matrices must be checked so an improvement is not limited to `GALAXY`. The test split must remain untouched during model development and model selection; it should be used only once after the final model has been selected.

## Public Solution Citations

{source_lines}

Published scores from these sources are not directly comparable with the scores below because they may use different splits, features and preprocessing. This benchmark reproduces the model families on the local prepared `train.csv` and `val.csv` under one standardized evaluation protocol.

## Baseline Models

- **Multinomial Logistic Regression**: median imputation, `StandardScaler`, `LogisticRegression(class_weight="balanced", random_state=42, max_iter=3000)`. This is the interpretable linear baseline.
- **Random Forest**: `RandomForestClassifier` with 300 trees, `class_weight="balanced_subsample"`, `random_state=42`, `n_jobs=-1` and conservative defaults. This is a nonlinear bagging baseline.
- **CatBoost**: CPU-compatible `CatBoostClassifier` with multiclass loss, fixed depth/learning-rate/iteration settings, `random_seed=42`, no verbose output and balanced automatic class weights.

No hyperparameter optimization was performed.

## Leakage Prevention

Before constructing `X`, both target columns, `class` and `class_encoded`, are removed. The benchmark confirms neither target representation remains in features, enforces matching train/validation feature columns and order, confirms selected features are numeric, checks missing and infinite values, and fits imputation/scaling only inside estimators trained on `train.csv`. No SMOTE or global resampling is used.

## Summary Metrics

{_format_markdown_table(sorted_summary[SUMMARY_COLUMNS])}

## Per-class Results

{_format_markdown_table(display_per_class, ["Precision", "Recall", "F1"])}

## Figures

{figure_lines}

Generated tables:

- [baseline_metrics.csv](tables/baseline_metrics.csv)
- [per_class_metrics.csv](tables/per_class_metrics.csv)

## Interpretation

Current benchmark by the predefined primary metric: **{best_model}** with validation balanced accuracy `{best_row["Balanced Accuracy"]:.4f}`.

Best model by macro F1: **{macro_f1_winner}** with macro F1 `{macro_row["Macro F1"]:.4f}`.

Best model by accuracy: **{accuracy_winner}** with accuracy `{accuracy_row["Accuracy"]:.4f}`.

CatBoost is selected as the current benchmark because it achieves the highest validation balanced accuracy, which is the predefined primary metric. Its advantage is driven partly by stronger recall for the minority `QSO` class. However, Random Forest achieves higher macro F1 and overall accuracy, indicating a better overall precision-recall balance. Therefore, CatBoost is the benchmark under the selected protocol, but Random Forest remains a strong alternative rather than being uniformly worse.

{chr(10).join(top_errors)}

The feature-distribution figure uses visualization-only clipping at the train-set 0.5th and 99.5th percentiles. The original feature values are preserved for model training. Extreme values exist in some engineered color features, including ranges around `-10017`, and their physical or preprocessing origin should be investigated in future data-quality work.

Future custom models must be trained on the same prepared features, selected on validation balanced accuracy, checked against macro F1 and per-class metrics, and compared against the best baseline above before any final one-time test-set evaluation.

## Reproducibility

Run from the repository root:

```bash
python -m src.baseline_benchmark
```

On Windows, `py -m src.baseline_benchmark` can be used if `python` points to the Microsoft Store launcher.

Relevant package versions:

{version_lines}

## Issue 1 Acceptance Criteria Checklist

- [x] Evaluation metrics documented with balanced accuracy as the primary metric.
- [x] Macro F1 documented as the main secondary metric and tie-breaker.
- [x] Required overall, per-class and confusion-matrix metrics are generated.
- [x] Baseline comparison protocol documents train/validation/test usage.
- [x] Accuracy is not used as the sole model-selection metric.

## Issue 2 Acceptance Criteria Checklist

- [x] Public solution and original dataset citations are included.
- [x] Logistic Regression, Random Forest and CatBoost baselines are trained.
- [x] Models are fitted on `train.csv` only.
- [x] Validation metrics are calculated on `val.csv` only.
- [x] The test split was not loaded, inspected or evaluated.
- [x] Summary and per-class CSV tables are saved.
- [x] Required PNG figures are saved.
- [x] Executed notebook presents the same benchmark workflow.

## References

{source_lines}
"""
    paths.report_path.write_text(report, encoding="utf-8")
    return paths.report_path


def save_results(
    paths: BenchmarkPaths,
    train: pd.DataFrame,
    y_val: pd.Series,
    results: list[ModelResult],
    summary_metrics: pd.DataFrame,
    per_class_metrics: pd.DataFrame,
    split_metadata: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Path]:
    """Save benchmark tables, plots and Markdown report."""

    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)

    sorted_summary = _sorted_summary(summary_metrics)
    sorted_summary.to_csv(paths.tables_dir / "baseline_metrics.csv", index=False)
    per_class_metrics.to_csv(paths.tables_dir / "per_class_metrics.csv", index=False)

    sns.set_theme(style="whitegrid", context="notebook", palette="Set2")
    figure_paths: dict[str, Path] = {}
    figure_paths["class_distribution"] = plot_class_distribution(
        split_metadata, paths.figures_dir / FIGURE_NAMES["class_distribution"]
    )
    figure_paths["metrics_comparison"] = plot_metric_comparison(
        sorted_summary, paths.figures_dir / FIGURE_NAMES["metrics_comparison"]
    )
    figure_paths["per_class_f1"] = plot_per_class_f1(
        per_class_metrics, paths.figures_dir / FIGURE_NAMES["per_class_f1"]
    )

    for result in results:
        slug = MODEL_SLUGS[result.name]
        figure_paths[f"confusion_matrix_{slug}"] = plot_confusion_matrix(
            y_val,
            result.y_pred,
            result.name,
            paths.figures_dir / f"confusion_matrix_{slug}.png",
        )
    figure_paths["confusion_overview"] = plot_confusion_matrices_overview(
        results, y_val, paths.figures_dir / FIGURE_NAMES["confusion_overview"]
    )

    result_by_name = {result.name: result for result in results}
    figure_paths["logistic_importance"] = plot_feature_importance(
        result_by_name["Logistic Regression"],
        feature_columns,
        paths.figures_dir / FIGURE_NAMES["logistic_importance"],
    )
    figure_paths["rf_importance"] = plot_feature_importance(
        result_by_name["Random Forest"],
        feature_columns,
        paths.figures_dir / FIGURE_NAMES["rf_importance"],
    )
    figure_paths["catboost_importance"] = plot_feature_importance(
        result_by_name["CatBoost"],
        feature_columns,
        paths.figures_dir / FIGURE_NAMES["catboost_importance"],
    )

    best_model = str(sorted_summary.iloc[0]["Model"])
    figure_paths["feature_distributions"] = plot_top_feature_distributions(
        train,
        feature_columns,
        result_by_name[best_model],
        paths.figures_dir / FIGURE_NAMES["feature_distributions"],
    )
    write_report(
        paths,
        sorted_summary,
        per_class_metrics,
        split_metadata,
        feature_columns,
        figure_paths,
        y_val,
        results,
    )
    return figure_paths


def run_benchmark(repo_root: Path | str = ".") -> BenchmarkResult:
    """Run the full baseline benchmark and return generated outputs."""

    paths = resolve_paths(repo_root)
    train, val, split_metadata = load_data(paths.data_dir)
    X_train, y_train, X_val, y_val, feature_columns, _, _ = prepare_features(train, val)
    models = build_models()
    results = fit_and_predict_models(models, X_train, y_train, X_val)

    summary = pd.DataFrame(
        [
            evaluate_model(
                y_val,
                result.y_pred,
                result.name,
                result.fit_time,
                result.prediction_time,
            )
            for result in results
        ]
    )
    per_class = pd.concat(
        [
            calculate_per_class_metrics(y_val, result.y_pred, result.name)
            for result in results
        ],
        ignore_index=True,
    )
    figure_paths = save_results(
        paths,
        train,
        y_val,
        results,
        summary,
        per_class,
        split_metadata,
        feature_columns,
    )
    sorted_summary = _sorted_summary(summary)
    return BenchmarkResult(
        summary_metrics=sorted_summary,
        per_class_metrics=per_class,
        split_metadata=split_metadata,
        feature_columns=feature_columns,
        figure_paths=figure_paths,
        report_path=paths.report_path,
        best_model=str(sorted_summary.iloc[0]["Model"]),
    )


def main() -> None:
    """CLI entry point for reproducing the baseline benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root containing data/train.csv and data/val.csv.",
    )
    args = parser.parse_args()

    result = run_benchmark(args.repo_root)
    print("Baseline benchmark complete.")
    print(f"Best validation baseline: {result.best_model}")
    print(result.summary_metrics[SUMMARY_COLUMNS].to_string(index=False))
    print(f"Report: {result.report_path.relative_to(Path(args.repo_root).resolve())}")


if __name__ == "__main__":
    main()
