# Final Validation Report

## Scope

The canonical reproduction workflow was executed on a clean host installation:

```bash
docker compose up --build --exit-code-from reproduce
```

The command completed with exit code 0. No host Python environment, private
credentials, GPU configuration, or path edits were required.

## Environment validation

- Python 3.11.0 was verified.
- All 38 packages in `requirements.lock.txt` were verified.
- The canonical platform was `linux/amd64` with CPU execution.
- The pinned container image and dependency installation completed successfully.

## Data validation

- Kaggle dataset version 1 downloaded through the documented public endpoint.
- The raw dataset SHA-256 matched the reference value.
- The generated train, validation, and test SHA-256 values all matched.
- The deterministic stratified 70/15/15 split completed with seed 42.

## Test validation

The container ran `python -m pytest -q`: 40 tests passed and 0 failed.

## Metric validation

Every metric was checked with absolute tolerance `1e-12`:

| Experiment           | Validation balanced accuracy | Validation macro F1 | Validation accuracy | Result      |
| -------------------- | ---------------------------: | ------------------: | ------------------: | ----------- |
| Logistic Regression  |           0.9476993243584669 |  0.9344956819867626 |  0.9409333333333333 | Exact match |
| Random Forest        |           0.9697020845241635 |  0.9732969664359739 |  0.9770666666666666 | Exact match |
| CatBoost             |           0.9723533081902743 |  0.9685890288006633 |  0.9723333333333334 | Exact match |
| Tabular ResNet       |           0.9693998067762958 |  0.9690497755767767 |  0.9728666666666667 | Exact match |
| Tuned Tabular ResNet |           0.9694004392953532 |  0.9704551328887030 |  0.9742000000000000 | Exact match |

The non-canonical native arm64 Random Forest drift remains documented in
`reports/reproducibility.md`; no relaxed tolerance is used in Docker.

## Final status

The previously reported Docker-runtime blocker is resolved. No Major or Severe
reproducibility weakness was observed in the clean-host canonical run.

## References

- Dataset provider, fedesoriano, “Stellar Classification Dataset - SDSS17”:
  https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
- Docker Compose documentation:
  https://docs.docker.com/compose/
- PyTorch reproducibility notes:
  https://docs.pytorch.org/docs/stable/notes/randomness.html
