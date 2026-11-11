# Baseline Benchmark and Evaluation Metrics

## Objective

This report defines the classification metrics for all baseline and future custom models, then benchmarks three public-solution-inspired model families on the prepared `train.csv` and `val.csv` splits.

## Dataset and Splits

The prepared train and validation files are stored in `data/` and use the verified target mapping `GALAXY -> 0`, `QSO -> 1`, `STAR -> 2`. The benchmark fits models only on `train.csv` and evaluates only on `val.csv`. The test split was not loaded or inspected. It remains reserved for the single final evaluation after model selection.

Features used by every baseline:

`alpha, delta, u, g, r, i, z, redshift, u_g, g_r, r_i, i_z, u_r, g_i`

Verified train and validation class distribution:

| Split   |   Rows | Class   |   Count | Percent   |
|:--------|-------:|:--------|--------:|:----------|
| train   |  70000 | GALAXY  |   41611 | 59.44%    |
| train   |  70000 | QSO     |   13273 | 18.96%    |
| train   |  70000 | STAR    |   15116 | 21.59%    |
| val     |  15000 | GALAXY  |    8917 | 59.45%    |
| val     |  15000 | QSO     |    2844 | 18.96%    |
| val     |  15000 | STAR    |    3239 | 21.59%    |

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

- [Logistic Regression and Random Forest project](https://github.com/wonhee3472/Stellar_Classification): Used as public precedent for comparing Logistic Regression and Random Forest on SDSS stellar classes.
- [Random Forest notebook](https://www.kaggle.com/code/leylamammadzada/stellar-classification-random-forest): Used as public precedent for a Random Forest stellar-classification baseline.
- [CatBoost notebook](https://www.kaggle.com/code/aerdem/stellar-classification-catboost-97-accuracy): Used as public precedent for a CatBoost multiclass stellar-classification baseline.
- [Original SDSS17 dataset](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17): Original Kaggle dataset used to prepare the local train, validation and test splits.

Published scores from these sources are not directly comparable with the scores below because they may use different splits, features and preprocessing. This benchmark reproduces the model families on the local prepared `train.csv` and `val.csv` under one standardized evaluation protocol.

## Baseline Models

- **Multinomial Logistic Regression**: median imputation, `StandardScaler`, `LogisticRegression(class_weight="balanced", random_state=42, max_iter=3000)`. This is the interpretable linear baseline.
- **Random Forest**: `RandomForestClassifier` with 300 trees, `class_weight="balanced_subsample"`, `random_state=42`, `n_jobs=-1` and conservative defaults. This is a nonlinear bagging baseline.
- **CatBoost**: CPU-compatible `CatBoostClassifier` with multiclass loss, fixed depth/learning-rate/iteration settings, `random_seed=42`, no verbose output and balanced automatic class weights.

No hyperparameter optimization was performed.

## Leakage Prevention

Before constructing `X`, both target columns, `class` and `class_encoded`, are removed. The benchmark confirms neither target representation remains in features, enforces matching train/validation feature columns and order, confirms selected features are numeric, checks missing and infinite values, and fits imputation/scaling only inside estimators trained on `train.csv`. No SMOTE or global resampling is used.

## Summary Metrics

| Model               |   Balanced Accuracy |   Macro F1 |   Accuracy |   Macro Precision |   Macro Recall |   Weighted F1 |   Fit Time (s) |   Prediction Time (s) |
|:--------------------|--------------------:|-----------:|-----------:|------------------:|---------------:|--------------:|---------------:|----------------------:|
| CatBoost            |              0.9724 |     0.9686 |     0.9723 |            0.965  |         0.9724 |        0.9723 |         7.5298 |                0.0172 |
| Random Forest       |              0.9697 |     0.9733 |     0.9771 |            0.9772 |         0.9697 |        0.9769 |        11.3982 |                0.1347 |
| Logistic Regression |              0.9477 |     0.9345 |     0.9409 |            0.923  |         0.9477 |        0.9411 |         1.0182 |                0.0062 |

## Per-class Results

| Model               | Class   |   Precision |   Recall |     F1 |   Support |
|:--------------------|:--------|------------:|---------:|-------:|----------:|
| Logistic Regression | GALAXY  |      0.9719 |   0.9277 | 0.9493 |      8917 |
| Logistic Regression | QSO     |      0.8784 |   0.9167 | 0.8971 |      2844 |
| Logistic Regression | STAR    |      0.9188 |   0.9988 | 0.9571 |      3239 |
| Random Forest       | GALAXY  |      0.9757 |   0.986  | 0.9808 |      8917 |
| Random Forest       | QSO     |      0.9655 |   0.9241 | 0.9443 |      2844 |
| Random Forest       | STAR    |      0.9905 |   0.9991 | 0.9948 |      3239 |
| CatBoost            | GALAXY  |      0.9827 |   0.9705 | 0.9766 |      8917 |
| CatBoost            | QSO     |      0.9354 |   0.9466 | 0.9409 |      2844 |
| CatBoost            | STAR    |      0.9768 |   1      | 0.9883 |      3239 |

## Figures

- [class_distribution.png](figures/class_distribution.png)
- [metrics_comparison.png](figures/metrics_comparison.png)
- [per_class_f1_comparison.png](figures/per_class_f1_comparison.png)
- [confusion_matrix_logistic_regression.png](figures/confusion_matrix_logistic_regression.png)
- [confusion_matrix_random_forest.png](figures/confusion_matrix_random_forest.png)
- [confusion_matrix_catboost.png](figures/confusion_matrix_catboost.png)
- [confusion_matrices_overview.png](figures/confusion_matrices_overview.png)
- [logistic_regression_coefficients.png](figures/logistic_regression_coefficients.png)
- [random_forest_feature_importance.png](figures/random_forest_feature_importance.png)
- [catboost_feature_importance.png](figures/catboost_feature_importance.png)
- [top_feature_distributions_by_class.png](figures/top_feature_distributions_by_class.png)

Generated tables:

- [baseline_metrics.csv](tables/baseline_metrics.csv)
- [per_class_metrics.csv](tables/per_class_metrics.csv)

## Interpretation

Current benchmark by the predefined primary metric: **CatBoost** with validation balanced accuracy `0.9724`.

Best model by macro F1: **Random Forest** with macro F1 `0.9733`.

Best model by accuracy: **Random Forest** with accuracy `0.9771`.

CatBoost is selected as the current benchmark because it achieves the highest validation balanced accuracy, which is the predefined primary metric. Its advantage is driven partly by stronger recall for the minority `QSO` class. However, Random Forest achieves higher macro F1 and overall accuracy, indicating a better overall precision-recall balance. Therefore, CatBoost is the benchmark under the selected protocol, but Random Forest remains a strong alternative rather than being uniformly worse.

- CatBoost: weakest validation F1 is `QSO` (0.9409). The most common off-diagonal error is true `GALAXY` predicted as `QSO` (186 rows, 2.09% of true `GALAXY`).
- Random Forest: weakest validation F1 is `QSO` (0.9443). The most common off-diagonal error is true `QSO` predicted as `GALAXY` (216 rows, 7.59% of true `QSO`).
- Logistic Regression: weakest validation F1 is `QSO` (0.8971). The most common off-diagonal error is true `GALAXY` predicted as `QSO` (359 rows, 4.03% of true `GALAXY`).

The feature-distribution figure uses visualization-only clipping at the train-set 0.5th and 99.5th percentiles. The original feature values are preserved for model training. Extreme values exist in some engineered color features, including ranges around `-10017`, and their physical or preprocessing origin should be investigated in future data-quality work.

**Caveat added after review (W1/W3 fix):** StandardScaler-based linear models and BatchNorm-based neural networks are disproportionately affected by sentinel values such as `-9999` compared to tree-based models (CatBoost, Random Forest). Tree models do not depend on feature scale, make splits on quantiles/percentiles rather than standardized moments, and are naturally robust to a handful of single extreme rows. Consequently, the pre-sentinel-cleaning neural baseline (trained before the review) understates achievable neural performance and was not on an equal preprocessing footing with the tree baselines. All neural training runs in the corrected workflow apply `clean_sentinel_values()` *before* imputation and scaling, which restores equal footing. See also `reports/corrections.md`.

Future custom models must be trained on the same prepared features, selected on validation balanced accuracy, checked against macro F1 and per-class metrics, and compared against the best baseline above before any final one-time test-set evaluation.

## Reproducibility

Run from the repository root:

```bash
python -m src.baseline_benchmark
```

On Windows, `py -m src.baseline_benchmark` can be used if `python` points to the Microsoft Store launcher.

Relevant package versions:

- `Python`: `3.11.2`
- `catboost`: `1.2.10`
- `matplotlib`: `3.10.8`
- `numpy`: `2.4.4`
- `pandas`: `3.0.3`
- `scikit-learn`: `1.9.0`
- `seaborn`: `0.13.2`

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

- Logistic Regression and Random Forest precedent:
  https://github.com/wonhee3472/Stellar_Classification
- Random Forest precedent:
  https://www.kaggle.com/code/leylamammadzada/stellar-classification-random-forest
- CatBoost precedent:
  https://www.kaggle.com/code/aerdem/stellar-classification-catboost-97-accuracy
- Dataset provider, fedesoriano, “Stellar Classification Dataset - SDSS17”:
  https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
