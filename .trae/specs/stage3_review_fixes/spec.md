# Stage 3 Review Fixes - Product Requirements Document

## Overview
- **Summary**: Fix four weaknesses (W1–W4) identified in the Stage 3 review of the Stellar Classification project. The defects span data preprocessing, reproducibility workflow, report consistency, and model-modification strength.
- **Purpose**: Eliminate or materially reduce the 33.75-point deduction by addressing every reported weakness with verifiable code and artifact changes.
- **Target Users**: Blue team and grading reviewers who re-run the canonical Docker reproduction path and audit report claims against project artifacts.

## Goals
1. Eliminate the -9999 sentinel-induced neural training instability. Normal objects must no longer be compressed into a tiny scaled range, and BatchNorm / training curves must not show collapse epochs.
2. Bring headline test results into the canonical reproduction path so `docker compose up` reproduces Table 6/Table 7 numbers.
3. Resolve every documented report-vs-artifact contradiction by fixing code paths and publishing a corrections document that lists and explains each fix.
4. Strengthen Requirement 3 evidence: add one verifiable architecture change (LayerNorm replacing BatchNorm in the residual network), a targeted single-factor ablation, and a tuned configuration whose measured improvement exceeds the project's own ~0.003 meaningful-difference threshold.

## Non-Goals
- Do not edit or regenerate `project.pdf`; the submitted PDF is immutable. Report corrections live in `reports/corrections.md` and in updated `reports/*.md` files where contradictions exist.
- Do not change the prepared train/val/test split CSVs or their recorded SHA-256 hashes. Sentinel cleaning happens as a post-load preprocessing step in every consuming module.
- Do not run or compare the cited public Kaggle/GitHub notebooks (W4 limitation); baseline model-family re-implementations remain sufficient for Requirement 1.
- Do not provide FT-Transformer/Hybrid checkpoints, histories, or configs that were not originally shipped; the corrections document explicitly marks those numbers as unverifiable context.
- Do not add confidence intervals, bootstrap resampling, or multi-seed repeated evaluation. The project's single-seed protocol is preserved; meaningful-difference discussion is updated only in prose.

## Background & Context
Review report (abbreviated):
- **W1 (Major, Quality of implementation)**: Raw features u, g, z contain sentinel -9999. `prepare_dataframe` builds colour features on top of them. `SimpleImputer("median")` and `StandardScaler` treat -9999 as a valid value, so 8 of 14 inputs have train-set means/stds around -10 000. TabularResNet uses `BatchNorm1d` in every block and the head; mini-batches containing the outlier row distort both batch and running statistics. Saved histories show 8/30 and 13/45 collapse epochs with BA dropping to 0.60–0.82. The initial-vs-tuned BA difference is 6×10⁻⁷, well below a single sample's ~4×10⁻⁵ impact. Tree models (RF, CatBoost) are essentially unaffected, so the neural-vs-boosting comparison is confounded.
- **W2 (Minor, Reproducibility)**: `scripts/reproduce.sh` never calls `src.evaluate_test`, so the headline test results (Table 6, abstract/conclusion) fall outside the documented workflow. `reports/.test_evaluated` ships and blocks re-runs unless the undocumented `--force` flag is passed. `reports/test_results.json` records Python 3.14.7 / NumPy 2.5.2 instead of the pinned 3.11.0 / 2.4.6 container. Neural checkpoints were trained on Apple MPS while `train_model.py` hard-codes CPU. `baseline_benchmark.ipynb` is called "executed" but contains no outputs. `reproducibility.md` refers to a non-existent `data/README.md`.
- **W3 (Major, Quality of the report)**: Sec 4.3 claims tuning cured late-epoch collapse, but the tuned run has more collapses (13/45 vs 8/30). Sec 1.2/7.2 claims all models are affected identically (false for tree-vs-BatchNorm-net). Sec 6.1 says "Docker is source of truth; no host Python used" vs test_results.json showing host Python 3.14.7. Sec 5.4 claims tuning gains generalize vs Sec 7.2's own ~0.003 meaningful-difference bar (gains = 0.0004 BA / 0.0027 macro F1). Sec 3.1/4.2 cites unreproducible FT-Transformer/Hybrid/rejected-variant numbers. Sec 6.6 calls the empty notebook "executed"; reproducibility.md Sec 6.9 says the test split "has not been used", contradicting Sec 5.
- **W4 (Minor, Minimal requirements compliance)**: Only one hand-picked tuned config vs one baseline config is compared. Five settings (LR, dropout, epochs, scheduler, clipping) change simultaneously with no grid/search/ablation. Architecture modification (LayerNorm, wider layer, extra head layer, weighted loss — Sec 4.2) has no code, config, or checkpoint. Measured primary-metric effect is 6×10⁻⁷. Requirement 1 cites own re-implementations "as precedent" and does not run or compare the referenced notebooks.

User trade-off decisions captured 2026-10-07:
- W1 sentinel cleaning = post-load preprocessing in every consumer module; split CSVs and SPLIT_SHA256 untouched.
- W2 reproduce.sh = call `evaluate_test --force`.
- W3 = fix code + add new `reports/corrections.md` (instead of rewriting existing markdowns wholesale).
- W4 = LayerNorm architecture swap (robust to outliers per W1 root cause) + one single-factor ablation config + a tuned config expected to show a real gain now that training is stable.

## Functional Requirements
- **FR-1**: Every module that loads split CSVs (train_model, evaluate_test, verify_results, baseline_benchmark) must replace sentinel -9999 with NaN in the raw-band columns u, g, z AND in the six derived colour columns (u_g, g_r, r_i, i_z, u_r, g_i) BEFORE fit/transform of any imputer or scaler. Conversion must be factored into a single reusable helper exported from `prepare_data.py` so every consumer applies identical logic.
- **FR-2**: `TabularResNetClassifier` and `TabularResidualBlock` must support a switchable normalization layer: default-initialize to BatchNorm1d (so legacy shipped checkpoints still load in legacy code paths) but accept a constructor flag that replaces every BatchNorm1d with LayerNorm along the same tensor axis. When fresh training runs are launched, both initial and tuned configs request LayerNorm.
- **FR-3**: `train_model.py` must expose three named configurations: `INITIAL_CONFIG` (baseline-ish, LayerNorm + same original epochs/LR/dropout ballpark), `ABLATION_NORM_CONFIG` (only swap: same as INITIAL_CONFIG but forced BatchNorm — isolates the normalization effect), `TUNED_CONFIG` (LayerNorm + a small, defensible set of changes that together push validation BA improvement above 0.003 vs INITIAL_CONFIG on a clean, non-collapsing training curve). History CSVs must be written for all three; the best-epoch checkpoint must be selected based on balanced accuracy with a non-zero `min_delta` and optional `early_stopping_patience`.
- **FR-4**: `scripts/reproduce.sh` must, after `src.verify_results` and before the anonymity audit, call `python -m src.evaluate_test --force --repo-root .` so the test tables, figures, and JSON are regenerated inside the container.
- **FR-5**: `reports/corrections.md` must enumerate each W3 contradiction exactly once, state the originally submitted claim, state the corrected claim or qualifying caveat, and link to the implementing code line or artifact. Additionally, update `reports/reproducibility.md` Sec 6.9 to say the test split IS evaluated by the canonical path and remove/qualify the `data/README.md` reference, and update `reports/baseline_benchmark.md` Sec 1.2 wording so it no longer asserts equal impact across all model families.
- **FR-6**: `evaluate_test.py` preprocessing must match `train_model.py` exactly — same sentinel-to-NaN columns, same imputer, same scaler fit on train after sentinel cleaning. The ResNet prediction helper must accept optional architecture keyword arguments (normalization type) so LayerNorm checkpoints load correctly.
- **FR-7**: `verify_results.py` and `reports/reference_metrics.json` must be updated to reflect the newly retrained checkpoints (initial, ablation, tuned — or at minimum initial and tuned) after sentinel cleaning, and the verifier must apply identical sentinel cleaning before evaluating any checkpoint. If three checkpoints ship, the verifier validates all three.

## Non-Functional Requirements
- **NFR-1 (Determinism)**: After fixes, a single Linux/amd64 Docker run of `scripts/reproduce.sh` must exit 0, produce non-erroneous history CSVs with zero collapse epochs (defined as val balanced_accuracy < 0.90 for any epoch after epoch 3), and write a `reports/test_results.json` recorded on Python 3.11.0.
- **NFR-2 (Backward-compatible loading of legacy checkpoints is NOT required for fresh evaluation)**: It is acceptable that legacy MPS-trained checkpoints no longer yield meaningful predictions on the cleaned/scaled pipeline. `verify_results.py` is updated to validate the newly trained CPU/LayerNorm checkpoints against new expected metrics.
- **NFR-3 (Unit tests)**: Existing pytest suite in `tests/` must continue to pass. New functionality (sentinel-cleaning helper, normalization flag) must have one or more dedicated tests.
- **NFR-4 (Clear attribution of tuning gains)**: In both code comments near `TUNED_CONFIG` and in `reports/corrections.md`, each changed parameter vs `INITIAL_CONFIG` is listed with a one-line rationale. The single-factor `ABLATION_NORM_CONFIG` documents the normalization-layer effect in isolation.
- **NFR-5 (No personal data)**: All newly added files and modified text must pass the existing `scripts/audit_anonymity.py` check.

## Constraints
- **Technical**: Python 3.11, PyTorch 2.14, scikit-learn 1.9, CatBoost 1.2.10 as locked. Split CSVs keep their hashes. Windows host development is allowed, but correctness is judged on the Linux/amd64 Docker path.
- **Business**: Submitted `project.pdf` cannot be altered. All corrections must live in markdown artifacts inside the repository.
- **Dependencies**: No new third-party libraries. Only stdlib + already-locked packages.

## Assumptions
- A clean CPU retraining with sentinel cleaning + LayerNorm will produce stable (no-collapse) training curves and a measurable tuned-vs-initial improvement exceeding 0.003 BA. If the observed delta after an honest run falls below 0.003, we document it honestly rather than overfitting a search to a noise target.
- The -9999 sentinel occurs in u, g, z only; derived colour columns inherit -9999 via subtraction or produce other extreme values when exactly one operand is -9999. The helper replaces any value == -9999 in the listed 9 columns (u, g, z, u_g, g_r, r_i, i_z, u_r, g_i) with NaN; this also catches derived values where both operands were -9999 (difference 0 is NOT a sentinel and is kept).
- 45–60 epochs on CPU at batch size 256 with 70k rows is acceptable runtime for a reviewer loop.

## Open Questions
- [x] W1 approach (confirmed: post-load cleaning helper from prepare_data; splits unchanged).
- [x] W2 evaluate_test integration (confirmed: `--force` in reproduce.sh).
- [x] W3 report remediation (confirmed: `reports/corrections.md` + targeted edits to reproducibility.md and baseline_benchmark.md for the most-visible contradictions).
- [x] W4 strengthening approach (confirmed: LayerNorm swap + one single-factor ablation config + stronger TUNED_CONFIG measured vs INITIAL_CONFIG).
- [ ] If after clean retraining the TUNED_CONFIG improvement over INITIAL_CONFIG still does not clear 0.003 BA, should we widen the search space slightly (e.g. try hidden_dim=384, num_blocks=5) or accept the honestly-measured smaller delta and document it in corrections.md? → Pending implementation evidence; decision point after Task 7 training run.

## Acceptance Criteria

### AC-1: Sentinel -9999 is treated as missing in every data consumer
- **Type**: `rule`
- **Given**: The project is on a machine where data/ splits have been prepared by `prepare_data.py` (original, sentinel-bearing CSVs).
- **When**: Any of `train_model.py`, `evaluate_test.py`, `verify_results.py`, or `baseline_benchmark.py` loads features and fits an imputer/scaler or feeds features into CatBoost/RF/LR.
- **Then**: Values equal to -9999 in columns u, g, z, u_g, g_r, r_i, i_z, u_r, g_i are converted to NaN BEFORE imputation/scaling/training, via one shared helper `clean_sentinel_values(df)` exported from `prepare_data.py`.
- **Pass Condition**: In a Python REPL, `clean_sentinel_values(pd.DataFrame({'u':[-9999, 19.1], 'u_g':[-9999, 1.0], 'r_i':[0.0, -0.5]}))` returns a frame whose first two rows for u and u_g are NaN while r_i values remain unchanged; and a grep of all four consumers shows exactly one call-site per file invoking the helper before imputer fit.
- **Evidence**: Unit test `tests/test_prepare_data.py::test_clean_sentinel_values` passes; code inspection links in `tasks.md` completion evidence.

### AC-2: TabularResNet supports a LayerNorm-variant architecture
- **Type**: `rule`
- **Given**: `src/model.py` is importable.
- **When**: A caller constructs `TabularResNetClassifier(use_layer_norm=True, ...)` and `TabularResNetClassifier(use_layer_norm=False, ...)` (or equivalent flag name).
- **Then**: The LayerNorm variant has zero `BatchNorm1d` submodules (checked via `any(isinstance(m, nn.BatchNorm1d) for m in model.modules())` == False) and uses `LayerNorm` at the same logical positions. The default/false variant matches the historical BatchNorm architecture so legacy state-dict keys still load.
- **Pass Condition**: Pytest `tests/test_model.py::test_layernorm_switch` passes.
- **Evidence**: Test output plus code link.

### AC-3: Three named configurations ship, one is a single-factor ablation, tuned history shows no collapse
- **Type**: `rule`
- **Given**: Source tree with fixes applied and data splits prepared.
- **When**: User runs `python -m src.train_model --experiment initial`, `--experiment ablation_norm`, `--experiment tuned` each to completion (or equivalent distinct invocation).
- **Then**: Three history CSVs are written, contain 14 input features, and every epoch after epoch 3 in all three histories has validation balanced_accuracy >= 0.90. TUNED_CONFIG best BA minus INITIAL_CONFIG best BA is reported in the console and persisted. ABLATION_NORM_CONFIG differs from INITIAL_CONFIG in exactly one documented parameter (the normalization flag) only.
- **Pass Condition**: Saved histories satisfy the >= 0.90 rule programmatically (check script or test); diff of INITIAL_CONFIG and ABLATION_NORM_CONFIG fields shows only the normalization-related flag differs.
- **Evidence**: History CSV excerpts and config diff block in completion evidence.

### AC-4: Canonical reproduce.sh runs evaluate_test and Docker environment metadata matches pinned versions
- **Type**: `rule`
- **Given**: A host with Docker + Compose installed and outbound HTTPS.
- **When**: Reviewer runs `docker compose up --build --exit-code-from reproduce` as documented.
- **Then**: The service exits 0; inside the resulting container or bind mount, `reports/test_results.json` exists and its `python` field starts with `3.11.`; `reports/.test_evaluated` timestamp is newer than the container start; scripts/reproduce.sh line list includes `evaluate_test --force` between verify_results and audit_anonymity.
- **Pass Condition**: Inspection of `scripts/reproduce.sh` shows the new line; if an actual Docker run is performed, captured log lines confirm evaluate_test ran and printed "=== FINAL TEST RESULTS ===".
- **Evidence**: Script diff link and (if environment allows) run log.

### AC-5: Report contradictions are enumerated and corrected, reproducibility and benchmark markdowns updated
- **Type**: `rule`
- **Given**: Reports directory.
- **When**: Reviewer opens `reports/corrections.md`.
- **Then**: The file lists each W3 contradiction with Original Claim / Corrected Claim / Implementing Link subsections. Additionally, `reports/reproducibility.md` Sec 6.9 no longer states "test split has not been used" and either replaces the `data/README.md` reference with a pointer to inline checksums or creates that README with the three split hashes; `reports/baseline_benchmark.md` Sec 1.2 (or equivalent section discussing extreme values) adds a caveat that StandardScaler + BatchNorm neural pipelines are disproportionately affected by sentinel values compared to tree baselines.
- **Pass Condition**: Grep for the offending phrases returns no matches or returns them only with clearly-marked corrections context; corrections.md file contains >= 7 enumerated items covering each W3 bullet.
- **Evidence**: File contents and link references.

### AC-6: Test evaluation preprocessing exactly matches training preprocessing, including sentinel cleaning
- **Type**: `rule`
- **Given**: evaluate_test.py source.
- **When**: Inspecting the evaluation code path.
- **Then**: The same sentinel-cleaning helper is applied to X_train and X_test before imputer/scaler fit/transform; the imputer and scaler classes and strategies are identical to train_model.py. ResNet checkpoint loading can specify normalization type if checkpoints are now LayerNorm.
- **Pass Condition**: Code inspection shows a call to `clean_sentinel_values` on train+test feature frames inside evaluate_test.py, and `resnet_predict` accepts architecture kwargs that are forwarded to `TabularResNetClassifier`.
- **Evidence**: Code links.

### AC-7: verify_results.py uses cleaned pipeline, new reference metrics, and new checkpoints validate
- **Type**: `rule`
- **Given**: Fresh retrained checkpoints exist and `reports/reference_metrics.json` is updated.
- **When**: `python -m src.verify_results` runs after baselines and training.
- **Then**: The verifier applies the sentinel cleaner, loads each new checkpoint (initial, tuned — and ablation if included), and exits 0 with exact-or-tolerance match against new expected metrics. The checksum list in reference_metrics.json contains the new checkpoint file hashes.
- **Pass Condition**: Command exits 0 and prints the "All metrics match" success line.
- **Evidence**: Captured stdout.

### AC-8: Overall neural stability quality (rubric)
- **Type**: `rubric`
- **Dimension**: Degree to which the fixed neural pipeline is free of the original sentinel-induced BatchNorm pathology and produces smooth, interpretable training curves.
- **Scale**: 1-5
- **Anchors**:
  - 1 = Same collapse rate or worse than original 8/30 + 13/45.
  - 3 = Collapses eliminated but curves remain very noisy (> ±0.015 BA swings between consecutive epochs after epoch 10).
  - 5 = Zero collapse epochs after epoch 3 in both initial and tuned runs; consecutive-epoch BA swings after epoch 10 stay within ±0.006, visually comparable to the stable CatBoost benchmark variance.
- **Pass Threshold**: >= 4
- **Evidence**: Training history CSV data and plots saved under reports/figures.

### AC-9: Reproducibility-path coverage quality (rubric)
- **Type**: `rubric`
- **Dimension**: Extent the documented one-command workflow now covers the headline metrics the report actually relies on.
- **Scale**: 1-5
- **Anchors**:
  - 1 = evaluate_test still outside the canonical script; review still needs undocumented flags.
  - 3 = evaluate_test is called in the script but with no guardrails or with mismatched preprocessing.
  - 5 = evaluate_test is called inside reproduce.sh with the correct force flag; preprocessing matches training exactly; generated test_results.json correctly records the pinned Python 3.11 version; Sec 6.9 wording is consistent.
- **Pass Threshold**: >= 4
- **Evidence**: reproduce.sh diff, test_results.json Python version, reproducibility.md Sec 6.9 diff.

### AC-10: Requirement 3 evidence strength (rubric)
- **Type**: `rubric`
- **Dimension**: Strength of evidence that the documented and reproducible model modification is real, targeted, and measured.
- **Scale**: 1-5
- **Anchors**:
  - 1 = Still only INITIAL vs TUNED with 5 parameters changing and no architecture or ablation.
  - 3 = Three configs exist with a single-factor ablation, but the measured tuned BA gain is still below the project's ~0.003 meaningful-difference bar with no honest discussion.
  - 5 = Three configs with clean single-factor LayerNorm ablation; each TUNED_CONFIG parameter change listed with rationale; measured BA gain either clears 0.003 or is honestly caveated with "below statistical bar" in corrections.md.
- **Pass Threshold**: >= 4
- **Evidence**: Config code block showing three configs and their diffs; history best-epoch deltas; corrections.md section.
