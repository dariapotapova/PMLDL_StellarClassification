# Stage 3 Review Fixes - Implementation Plan

## Task 1: Add sentinel-cleaning helper to prepare_data.py and tests
- **Status**: `pending`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - Add a new top-level function `clean_sentinel_values(df: pd.DataFrame) -> pd.DataFrame` (and/or in-place variant) to `src/prepare_data.py`. The function replaces any value equal to -9999 in the columns `['u','g','z','u_g','g_r','r_i','i_z','u_r','g_i']` with `np.nan`; all other columns and values are untouched.
  - Export the function at module level (it is called from every consumer).
  - Add a unit test `test_clean_sentinel_values` under `tests/` (either a new `test_prepare_data.py` or appended to an existing test file) covering: original -9999 in u/g/z become NaN; derived -9999 in u_g/etc become NaN; a value of 0 in a derived column (e.g. r_i) is NOT replaced; a non-listed numeric column (e.g. redshift) with -9999 is NOT replaced.
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `rule` TR-1.1: `pytest -q tests/test_prepare_data.py::test_clean_sentinel_values` exits with code 0.
  - `rule` TR-1.2: `clean_sentinel_values(pd.DataFrame({'u':[-9999], 'redshift':[-9999], 'r_i':[0.0]}))` returns NaN in u only (checked via assert in test body).

## Task 2: Refactor all four data consumers to apply sentinel cleaning before training/evaluation
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - In `src/train_model.py`: import `clean_sentinel_values` from `src.prepare_data`; apply it to `X_train` and `X_val` (as DataFrames, not numpy) immediately after the `train.drop(TARGET_COLUMNS)` / `val.drop(...)` lines and BEFORE the imputer/scaler fit/transform.
  - In `src/evaluate_test.py`: same — import the helper, apply to `X_train` (from prepare_features or loaded directly) and to `X_test` BEFORE the `SimpleImputer` / `StandardScaler` fit/transform on lines 187-189.
  - In `src/verify_results.py`: import helper, apply to train and val frames in `neural_metrics()` BEFORE `imputer.fit_transform(...)`.
  - In `src/baseline_benchmark.py`: find the `prepare_features()` function and apply `clean_sentinel_values()` to the train and val feature DataFrames before building pipelines or fitting estimators. Ensure LogisticRegression's StandardScaler pipeline and RF/CatBoost inputs are clean.
- **Acceptance Criteria Addressed**: AC-1, AC-6
- **Test Requirements**:
  - `rule` TR-2.1: Grep for `clean_sentinel_values(` in each of the 4 files returns exactly one or more match per file, all appearing BEFORE any `SimpleImputer` / `StandardScaler` / `.fit(` call operating on those frames.
  - `rule` TR-2.2: `pytest -q` exits with code 0 on the existing suite (baseline model metrics will shift slightly on real data but tests that don't depend on exact numerical outputs still pass).

## Task 3: Add LayerNorm/ BatchNorm switch to TabularResidualBlock and TabularResNetClassifier
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 1 (independent of W1; ordering with Task 2 is arbitrary)
- **Description**:
  - In `src/model.py`, add a constructor flag `use_layer_norm: bool = False` to both `TabularResidualBlock` and `TabularResNetClassifier`.
  - When `use_layer_norm=False` (default), the architecture is byte-for-byte identical to the historical BatchNorm implementation so old state-dict keys still map.
  - When `use_layer_norm=True`, replace each `nn.BatchNorm1d(hidden_dim)` with `nn.LayerNorm(hidden_dim)` at the same sequence positions (both inside each residual block and in the classifier head).
  - Add a unit test `test_layernorm_switch` in `tests/test_model.py`: two models with equal seeds/random state, one with True and one with False; verify that `any(isinstance(m, nn.BatchNorm1d) for m in ln_model.modules())` is False for the LN model and True for the BN model; verify both models produce output of shape `[batch, 3]` on random inputs.
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `rule` TR-3.1: `pytest -q tests/test_model.py::test_layernorm_switch` exits with code 0.
  - `rule` TR-3.2: Default model `TabularResNetClassifier()` has at least 5 BatchNorm modules (4 blocks + 1 head); LN variant has 0 BatchNorm modules and >= 5 LayerNorm modules.

## Task 4: Add three training configs (initial / ablation_norm / tuned) to train_model.py and wire up LayerNorm
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 2, Task 3
- **Description**:
  - Replace the existing two-config scheme with three named frozen dataclass instances:
    1. `INITIAL_CONFIG`: LayerNorm enabled, epochs 30, LR ~1e-3 or 7e-4, dropout 0.10, no scheduler, no clipping — deliberately the "old but fixed-preprocessing + new-architecture baseline".
    2. `ABLATION_NORM_CONFIG`: identical to INITIAL_CONFIG in every field except it forces `use_layer_norm=False` (BatchNorm) on the model. Used to isolate and measure the BatchNorm→LayerNorm effect in isolation (single-factor ablation).
    3. `TUNED_CONFIG`: LayerNorm enabled, epochs 45–60, moderately better settings — e.g. LR 5e-4 or 7e-4 with scheduler, dropout 0.05, gradient clip 1.0, early stopping patience 10, `min_delta=1e-4`. Each changed parameter vs INITIAL_CONFIG is listed with a one-line rationale comment in code.
  - Update the `--experiment` argparse choices: `initial`, `ablation_norm`, `tuned`. Default = `tuned`.
  - Pass `use_layer_norm=(config.use_layer_norm if hasattr else True)` to the model constructor so the correct normalization is used.
  - Update the `file_prefix` mapping: `tabular_resnet`, `tabular_resnet_ablation_norm`, `tabular_resnet_tuned` so three distinct checkpoints and history CSVs are written.
  - Ensure at the end of training, the script also prints a line: `Best BA delta (tuned - initial) = X.XXXX` (when experiment == tuned and both CSVs exist) or otherwise just stores the per-experiment best BA clearly.
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `rule` TR-4.1: Diff of INITIAL_CONFIG fields vs ABLATION_NORM_CONFIG fields shows exactly one logical difference: the normalization flag / use_layer_norm attribute.
  - `rule` TR-4.2: Argparse --help lists three experiments; passing an invalid value produces a clear error.
  - `rubric` TR-4.3: Config clarity and documentation; scale 1-5; 1=fields undocumented/confusing, 3=some rationale comments, 5=each tuned change has one-line rationale and ablation_norm is clearly commented as a single-factor control; threshold >= 4; evidence = code inspection.

## Task 5: Update reproduce.sh to run evaluate_test --force, and fix reproducibility.md references
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 4 (logically before training runs, but textually independent)
- **Description**:
  - In `scripts/reproduce.sh`, insert a new line between `python -m src.verify_results` and `python scripts/audit_anonymity.py`:
    ```
    python -m src.evaluate_test --force
    ```
  - In `reports/reproducibility.md` Sec 6.9:
    - Replace the sentence "The reserved test split has not been used for a final reported evaluation." with a sentence noting the test split IS evaluated by the canonical workflow via `evaluate_test --force` and results are in `reports/tables/test_metrics.csv` + `reports/test_results.json`.
    - Either: (a) remove the reference to `data/README.md` in Sec 6.3 and inline the three split SHA-256 values that currently appear only in `src/prepare_data.py` SPLIT_SHA256, OR (b) create `data/README.md` with those three hashes (and .gitkeep the data directory if needed). Preferred: (a) inline into reproducibility.md so no new tracked file under data/ is required.
  - In `reports/reproducibility.md` Sec 6.6 wording: change the claim that `baseline_benchmark.ipynb` is "executed" to "shipped as a reference notebook; its outputs are mirrored in reports/baseline_benchmark.md", since the ipynb currently contains no outputs.
- **Acceptance Criteria Addressed**: AC-4, AC-5
- **Test Requirements**:
  - `rule` TR-5.1: Grep scripts/reproduce.sh finds `evaluate_test` and `--force` on the same line, appearing after `verify_results` and before `audit_anonymity`.
  - `rule` TR-5.2: Grep reports/reproducibility.md for "has not been used" returns 0 matches.
  - `rule` TR-5.3: Grep reports/reproducibility.md for "data/README.md" returns 0 matches, OR the file `data/README.md` exists and contains three SPLIT_SHA256 values.

## Task 6: Fix evaluate_test.py architecture loading + preprocessing match, and fix baseline_benchmark.md wording
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 2 (sentinel cleaning in evaluate_test), Task 3 (LN/BN flag exists)
- **Description**:
  - In `src/evaluate_test.py`, extend `resnet_predict(checkpoint, X_scaled, **kwargs)` so that any `**kwargs` are forwarded to `TabularResNetClassifier(**kwargs)`. Update the RESNET_RUNS dict to carry architecture kwargs alongside each checkpoint filename, so both LN and BN checkpoints can load if they coexist. (With our three-config plan, initial and tuned are LN; ablation_norm is BN.)
  - Verify evaluate_test.py's preprocessing exactly matches train_model.py's: same columns passed to clean_sentinel_values, same imputer strategy=median, same StandardScaler fit on train-only after cleaning.
  - In `reports/baseline_benchmark.md` Sec 1.2 / the paragraph that mentions extreme values at line 127 (currently "Extreme values exist in some engineered color features..."): append a caveat paragraph: "Caveat added after review: StandardScaler-based linear models and BatchNorm-based neural networks are disproportionately affected by sentinel values such as -9999 compared to tree-based models (CatBoost, Random Forest), which do not depend on feature scale and are robust to single extreme rows. Consequently, the pre-sentinel-cleaning neural baseline understates achievable neural performance and was not on an equal footing with the tree baselines."
- **Acceptance Criteria Addressed**: AC-5, AC-6
- **Test Requirements**:
  - `rule` TR-6.1: `resnet_predict` signature accepts **kwargs and forwards them to TabularResNetClassifier; a unit test or ad-hoc call shows kwargs propagate.
  - `rule` TR-6.2: evaluate_test.py applies clean_sentinel_values to both train and test frames before imputer/scaler.
  - `rule` TR-6.3: baseline_benchmark.md contains the caveat substring about StandardScaler/BatchNorm disproportionate impact.

## Task 7: Run fresh clean CPU training for initial/ablation_norm/tuned experiments and inspect histories
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 1–Task 4, Task 6 (preprocessing and architecture code complete)
- **Description**:
  - Run `python -m src.train_model --experiment initial` → saves models/tabular_resnet.pt + history.
  - Run `python -m src.train_model --experiment ablation_norm` → saves ablation checkpoint + history.
  - Run `python -m src.train_model --experiment tuned` → saves tuned checkpoint + history.
  - Load all three histories; verify zero collapse epochs after epoch 3 (BA >= 0.90). Compute BA delta tuned-initial. If the delta is clearly positive and robust (> ~0.003), accept. If the delta is near-zero or negative, make one small, well-motivated adjustment to TUNED_CONFIG (e.g. wider hidden_dim 320, or LR warm-up not required — but keep number of ad-hoc rounds <= 2) and re-run tuned only. Record the change. Do NOT over-search. If after <= 2 honest rounds the delta still does not clear 0.003, document honestly and proceed.
  - Save the produced histories and plots. They overwrite the shipped ones.
- **Acceptance Criteria Addressed**: AC-3, AC-8, AC-10
- **Test Requirements**:
  - `rule` TR-7.1: A programmatic check on the three history CSVs finds no epoch>3 with BA < 0.90 in the initial and tuned runs. (Ablation_norm is allowed some residual collapse because it deliberately retains BatchNorm — but ideally fewer than before.)
  - `rubric` TR-7.2: Neural curve smoothness assessed from plots; scale 1-5 as per AC-8; threshold >= 4; evidence = saved BA plots.
  - `rubric` TR-7.3: Tuned gain quality vs. reporting honesty; scale 1-5 (1=delta tiny but overclaimed, 3=delta honest but <0.003 with no caveat, 5=either delta>=0.003 OR delta smaller with explicit caveat); threshold >= 4; evidence = console output + corrections.md section.

## Task 8: Update reference_metrics.json, verify_results.py (3 checkpoints), and re-run whole pipeline once
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 7 (new checkpoints exist)
- **Description**:
  - Compute SHA-256 for each new checkpoint (tabluar_resnet.pt, tabular_resnet_ablation_norm.pt if retained, tabular_resnet_tuned.pt) via `prepare_data.sha256` or Python.
  - Edit `reports/reference_metrics.json`: update checkpoints object with new hashes; update validation_metrics with the newly observed initial/ablation/tuned BA, macro F1, accuracy; keep baseline metrics as they will be regenerated and verified to the same strict tolerance (or update tolerances if sentinel cleaning shifted CatBoost/RF — but since trees are robust, we expect negligible shift and the verifier will tell us).
  - Edit `src/verify_results.py`:
    - Apply clean_sentinel_values in neural_metrics (already done in Task 2).
    - Update RESNET runs list in the verifier to evaluate initial, ablation, and tuned checkpoints, each with its correct dropout and use_layer_norm flag.
    - Use the `weights_only=False` or default `torch.load` if needed (current code uses weights_only=True; must match how models are saved in train_model.py).
  - Run full local pipeline once in order: prepare_data → baseline_benchmark → train_model all three → verify_results → evaluate_test --force → audit_anonymity. Fix any mismatches iteratively. Capture final successful verify_results stdout.
- **Acceptance Criteria Addressed**: AC-7, AC-9
- **Test Requirements**:
  - `rule` TR-8.1: `python -m src.verify_results` exits 0 and prints success line.
  - `rule` TR-8.2: `python -m src.evaluate_test --force` runs to completion without error and writes reports/test_results.json with `python` field == 3.11.x (if run in Docker) or whatever the current env Python is locally; we re-check in Docker-centric evidence in Task 9.
  - `rubric` TR-8.3: Breadth of verifier coverage; all three neural configs + baselines all covered; scale 1-5; threshold >= 4.

## Task 9: Write reports/corrections.md and final targeted edits to report markdowns
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 5, Task 6, Task 8 (all implementing evidence exists before the corrections doc can cite it)
- **Description**:
  - Create `reports/corrections.md` with sections mirroring the W1–W4 review findings. For every W3 bullet (there are 7–8 listed contradictions):
    - Subsection: "Claim X (Section Y.Z)"
    - "Original claim" — quote or paraphrase the PDF/markdown sentence.
    - "Correction / Caveat" — the corrected statement.
    - "Implementing evidence" — absolute `file://` link to the specific code lines, new config, or artifact that fixes or substantiates it.
  - Also add sections for W1, W2, W4 corrections even though they are not "report-contradictions", so the document is a one-stop shop for review-response fixes. E.g. W1 section lists the new helper and where it is called; W2 section shows reproduce.sh line; W4 lists the 3 configs and reports the measured BA delta with an honest statement about whether it clears the 0.003 bar.
  - Small final pass through reports/reproducibility.md and reports/baseline_benchmark.md to make sure edits from Tasks 5 and 6 are still consistent and cover every W3 bullet. Specifically for W3 bullets that cannot be completely resolved (e.g. FT-Transformer/Hybrid numbers have no checkpoints): add a note in corrections.md that these were contextual architecture trials and the model-selection decision (Tabular ResNet) is the only reproducible one.
- **Acceptance Criteria Addressed**: AC-5, AC-9, AC-10
- **Test Requirements**:
  - `rule` TR-9.1: corrections.md file exists and contains >= 7 enumerated "Original claim / Correction" pairs (one per W3 bullet).
  - `rule` TR-9.2: Each link in corrections.md uses the `[text](file:///C:/abs/path#Lx-Ly)` format and points to an existing file.
  - `rubric` TR-9.3: Overall corrections quality and candidness; scale 1-5 (1=deflective/omits bullets, 3=covers bullets but vague, 5=enumerates every W3 bullet honestly with links and candid discussion); threshold >= 4.

## Task 10: Re-run pytest suite and final verify_results + evaluate_test end-to-end pass, capture evidence
- **Status**: `pending`
- **Priority**: medium
- **Depends On**: Task 1–Task 9 (all)
- **Description**:
  - Run `pytest -q`; confirm no regressions.
  - Run `python -m src.verify_results` → capture stdout showing all metrics pass.
  - Run `python -m src.evaluate_test --force` → capture stdout.
  - Run `python scripts/audit_anonymity.py` if available or check no new personal data was added.
  - In reproduce.sh order, do one final manual sequential pass of the script lines to confirm they still compose.
  - If a Docker build is feasible in the environment, run it; if not, note in review.md that Docker confirmation is pending reviewer execution and all local steps pass.
- **Acceptance Criteria Addressed**: AC-1 through AC-10 (final integration gate)
- **Test Requirements**:
  - `rule` TR-10.1: `pytest -q` exit code 0.
  - `rule` TR-10.2: verify_results exit 0, evaluate_test exit 0.
  - `rubric` TR-10.3: Overall integration fidelity; scale 1-5; threshold >= 4; evidence = command output captures.
