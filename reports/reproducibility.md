# 6. Reproducibility

## 6.1 Canonical environment

The canonical environment is Docker using `python:3.11.0-slim-bullseye` pinned
to manifest digest `sha256:b5ee5360b357dcd0ab2e1cc5e22fa589b6734bbd55776df6b1700994febf1527`, Python
3.11.0 on `linux/amd64`, and CPU execution. `requirements.txt` pins every direct
Python dependency with `==`, while `requirements.lock.txt` pins the complete
resolved environment. The Docker build runs `pip check`, and the entrypoint
verifies Python and every locked distribution before work. The verified direct
versions are CatBoost 1.2.10, certifi 2026.7.22, Matplotlib 3.11.1, NumPy 2.4.6, pandas 3.0.5,
pytest 9.1.1, scikit-learn 1.9.0, seaborn 0.13.2, tabulate 0.10.0, and PyTorch
2.14.0. Docker is the source of truth; no host Python installation is used.
The fixed architecture selects widely available Linux wheels and consistent
numerical behavior; Docker Desktop provides emulation on non-amd64 hosts.

## 6.2 Randomness and determinism

The project-wide seed is 42. `src/reproducibility.py` controls Python
`random`, `PYTHONHASHSEED`, NumPy, PyTorch CPU and CUDA seeds, deterministic
PyTorch algorithms, cuDNN settings, and the DataLoader generator/worker seeds.
The split, scikit-learn estimators, and CatBoost also use seed 42. The canonical
container fixes numerical thread counts to one and uses CPU only.

The saved neural checkpoints were historically trained on Apple MPS. Their
reported validation predictions and metrics have been verified on CPU, but a
fresh CPU retraining is not claimed to be bitwise identical to the historical
MPS optimization trajectory. The canonical run therefore retrains deterministic
baselines and verifies neural checkpoints. Every reported metric, including
Random Forest, is compared with strict absolute tolerance `1e-12` in the
canonical linux/amd64 container. Timing columns are excluded because wall-clock
time is not a scientific result.

A non-canonical native arm64 run produced a `0.0005435868` Random Forest
balanced-accuracy difference from the stored historical result. This observed
platform drift is documented but is not accepted by the canonical verifier and
does not justify a relaxed Docker tolerance. The clean linux/amd64 Docker run
reproduced the historical Random Forest metrics exactly.

## 6.3 Data access

The source is version 1 of the public “Stellar Classification Dataset -
SDSS17,” maintained on Kaggle by fedesoriano and derived from Sloan Digital Sky
Survey DR17. The canonical script downloads the versioned archive from the
official Kaggle API without credentials. It fails clearly if the resource is
unavailable or if the extracted CSV SHA-256 differs from
`faefcad9b950309282024dd351289ee14ae30f65a7b2ef32c9fa442da9dbe585`.
All generated hashes are listed in `data/README.md` and
`reports/reference_metrics.json`. No private drive or local path is used.

## 6.4 Data preparation

`src/prepare_data.py` reproduces the original notebook procedure. It removes
nine identifier/observation metadata columns, creates `u_g`, `g_r`, `r_i`,
`i_z`, `u_r`, and `g_i` colour differences, maps `GALAXY/QSO/STAR` to `0/1/2`,
and makes stratified 70/15/15 train/validation/test splits with seed 42. The
result contains 70,000/15,000/15,000 rows and 14 numerical model features. All
three generated files must match their recorded SHA-256 values.

## 6.5 Docker reproduction

Run exactly:

```bash
docker compose up --build --exit-code-from reproduce
```

Compose propagates the reproduction service exit code. The workflow installs
dependencies, downloads and prepares data, runs tests, retrains baselines,
evaluates saved neural checkpoints, verifies metrics, audits anonymity, and
writes `reproduction_outputs/verified_metrics.json` on the host.

## 6.6 Reproducing baselines

The canonical workflow runs `python -m src.baseline_benchmark`. Logistic
Regression, Random Forest, and CatBoost are fitted on `train.csv` and evaluated
on `val.csv`. Stable metrics appear in `reports/tables/baseline_metrics.csv` and
the canonical consolidated JSON output. Figures and per-class metrics are
regenerated in `reports/` inside the container.

## 6.7 Reproducing the proposed model

The proposed Tabular ResNet training implementation is `src/train_model.py`.
The historical proposed checkpoint is `models/tabular_resnet.pt`; the canonical
workflow checks its hash and reevaluates it on the recreated validation split.
This is the reproducible verification path for the reported historical result.
To run the initial 30-epoch CPU training configuration, use:

```bash
docker compose run --rm reproduce python -m src.train_model --experiment initial
```

## 6.8 Reproducing the improved/tuned model

The tuned configuration in `src/train_model.py` uses learning rate `7e-4`,
dropout 0.05, gradient clipping, and `ReduceLROnPlateau`. To launch a fresh,
deterministic CPU tuned run in the container, use:

```bash
docker compose run --rm reproduce python -m src.train_model --experiment tuned
```

The canonical one-command workflow instead verifies the selected checkpoint
`models/tabular_resnet_tuned.pt`, because the published artifact was trained on
MPS and fresh CPU optimization is not expected to be bitwise identical.

## 6.9 Expected results

| Experiment | Validation balanced accuracy | Validation macro F1 | Validation accuracy |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.9476993243584669 | 0.9344956819867626 | 0.9409333333333333 |
| Random Forest | 0.9697020845241635 | 0.9732969664359739 | 0.9770666666666666 |
| CatBoost | 0.9723533081902743 | 0.9685890288006633 | 0.9723333333333334 |
| Tabular ResNet | 0.9693998067762958 | 0.9690497755767767 | 0.9728666666666667 |
| Tuned Tabular ResNet | 0.9694004392953532 | 0.9704551328887030 | 0.9742000000000000 |

These are validation results. The reserved test split has not been used for a
final reported evaluation.

## 6.10 Output artifacts

The reviewer-facing machine-readable output is
`reproduction_outputs/verified_metrics.json`. Baseline tables and figures are
regenerated under `reports/` in the container. Neural training writes its
checkpoint to `models/tabular_resnet_tuned.pt`, history to `reports/training/`,
and plots to `reports/figures/` inside the run container.

## 6.11 Validation status and known limitations

The canonical command was executed on a clean host installation and completed
with exit code 0. It verified Python 3.11.0, all 38 locked packages, raw and
split checksums, 40 passing tests, exact baseline metrics, both neural
checkpoints, and the anonymity audit. The complete record is in
`reports/final_validation.md`.

The canonical run requires outbound HTTPS access to Kaggle. It fails rather
than silently continuing if the pinned public resource changes or becomes
unavailable. Native non-canonical arm64 execution may exhibit the Random Forest
drift described above. Historical FT-Transformer, Hybrid FT-Transformer, and
rejected wider/class-weighted variant values have incomplete machine-readable
training histories; they remain contextual architecture trials, not
authoritative final metrics.

## References

- Dataset provider, fedesoriano, “Stellar Classification Dataset - SDSS17”:
  https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
- Sloan Digital Sky Survey DR17: https://www.sdss.org/dr17/
- Kaggle API documentation: https://www.kaggle.com/docs/api
- PyTorch reproducibility notes:
  https://docs.pytorch.org/docs/stable/notes/randomness.html
- Docker Compose file reference:
  https://docs.docker.com/compose/compose-file/