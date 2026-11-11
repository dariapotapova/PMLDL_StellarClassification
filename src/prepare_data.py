"""Download and reproduce the prepared SDSS17 data splits.

This module preserves the preprocessing originally recorded in
``stellar_class_data.ipynb``: metadata removal, six colour differences,
alphabetical target encoding, and two stratified splits with seed 42.

References
----------
Dataset page and attribution:
https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

Official Kaggle API documentation:
https://www.kaggle.com/docs/api

scikit-learn train_test_split documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import ssl
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import certifi
import pandas as pd
from sklearn.model_selection import train_test_split

from src.reproducibility import PROJECT_SEED

DATASET_VERSION = 1
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "fedesoriano/stellar-classification-dataset-sdss17?datasetVersionNumber=1"
)
RAW_FILENAME = "star_classification.csv"
RAW_SHA256 = "faefcad9b950309282024dd351289ee14ae30f65a7b2ef32c9fa442da9dbe585"
SPLIT_SHA256 = {
    "train.csv": "f0659a57c02afc18d7860c886233a30719b43d5cdc28615307b0161a4c665d21",
    "val.csv": "aa84702f24b2cf1b72f54aaac489a2090748dce064737f7a5d3a0abf5c855c6d",
    "test.csv": "33a4c0a9a103b9ced0804103bd45bb02a1bf72d24487d785ad1ad376dd9a80a0",
}
METADATA_COLUMNS = [
    "obj_ID",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "spec_obj_ID",
    "MJD",
    "fiber_ID",
    "plate",
]
LABEL_MAPPING = {"GALAXY": 0, "QSO": 1, "STAR": 2}
COLOUR_FEATURES = {
    "u_g": ("u", "g"),
    "g_r": ("g", "r"),
    "r_i": ("r", "i"),
    "i_z": ("i", "z"),
    "u_r": ("u", "r"),
    "g_i": ("g", "i"),
}
SENTINEL_VALUE = -9999
SENTINEL_COLUMNS = ["u", "g", "z", "u_g", "g_r", "r_i", "i_z", "u_r", "g_i"]


def clean_sentinel_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the SDSS missing-value sentinel -9999 with NaN in raw and derived bands.

    Operates on a copy; the input frame is never mutated in place. Only the
    columns listed in :data:`SENTINEL_COLUMNS` are inspected; any other
    column (including ``redshift``, identifier metadata, or the target) is
    passed through untouched. A value of exactly zero in a derived colour
    (e.g. ``r_i == 0`` when both bands are identical on the same object)
    is **not** treated as a sentinel, which distinguishes a real
    measurement from the ``-9999 - (-9999) = 0`` artefact-free case.
    """

    result = df.copy()
    for column in SENTINEL_COLUMNS:
        if column in result.columns:
            result.loc[result[column] == SENTINEL_VALUE, column] = pd.NA
    return result


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw_data(data_dir: Path) -> Path:
    """Download dataset version 1 and verify the extracted raw CSV."""

    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / RAW_FILENAME
    if raw_path.exists() and sha256(raw_path) == RAW_SHA256:
        return raw_path

    with tempfile.TemporaryDirectory(prefix="stellar-data-") as temporary:
        archive_path = Path(temporary) / "dataset.zip"
        try:
            tls_context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(
                DATASET_URL, timeout=120, context=tls_context
            ) as response:
                with archive_path.open("wb") as archive:
                    shutil.copyfileobj(response, archive)
        except Exception as error:
            raise RuntimeError(
                "Could not download immutable Kaggle dataset version 1 from the "
                f"documented public URL: {error}"
            ) from error
        with zipfile.ZipFile(archive_path) as archive:
            if RAW_FILENAME not in archive.namelist():
                raise RuntimeError(f"Downloaded archive does not contain {RAW_FILENAME}.")
            with archive.open(RAW_FILENAME) as source, raw_path.open("wb") as target:
                shutil.copyfileobj(source, target)

    actual = sha256(raw_path)
    if actual != RAW_SHA256:
        raw_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Raw data checksum mismatch: expected {RAW_SHA256}, received {actual}."
        )
    return raw_path


def prepare_dataframe(raw_path: Path) -> pd.DataFrame:
    """Apply the historically recorded feature and target transformations."""

    frame = pd.read_csv(raw_path).drop(columns=METADATA_COLUMNS, errors="ignore")
    for output, (left, right) in COLOUR_FEATURES.items():
        frame[output] = frame[left] - frame[right]
    frame["class_encoded"] = frame["class"].map(LABEL_MAPPING)
    if frame["class_encoded"].isna().any():
        raise ValueError("Raw data contains an unknown target class.")
    frame["class_encoded"] = frame["class_encoded"].astype(int)
    return frame


def split_dataframe(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create the original stratified 70/15/15 split using seed 42."""

    features = frame.drop(columns=["class_encoded"])
    target = frame["class_encoded"]
    train_x, temporary_x, train_y, temporary_y = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=PROJECT_SEED,
        stratify=target,
    )
    val_x, test_x, val_y, test_y = train_test_split(
        temporary_x,
        temporary_y,
        test_size=0.50,
        random_state=PROJECT_SEED,
        stratify=temporary_y,
    )
    result = {}
    for name, split_x, split_y in (
        ("train", train_x, train_y),
        ("val", val_x, val_y),
        ("test", test_x, test_y),
    ):
        output = split_x.copy()
        output["class_encoded"] = split_y
        result[name] = output
    return result


def write_and_verify_splits(splits: dict[str, pd.DataFrame], data_dir: Path) -> None:
    """Write split CSVs and require their known hashes."""

    for name, frame in splits.items():
        path = data_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        actual = sha256(path)
        expected = SPLIT_SHA256[path.name]
        if actual != expected:
            raise RuntimeError(
                f"Prepared {path.name} checksum mismatch: expected {expected}, "
                f"received {actual}."
            )


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    raw_path = download_raw_data(args.data_dir)
    splits = split_dataframe(prepare_dataframe(raw_path))
    write_and_verify_splits(splits, args.data_dir)
    print(f"Verified raw dataset version {DATASET_VERSION}: {RAW_SHA256}")
    for filename, checksum in SPLIT_SHA256.items():
        print(f"Verified {filename}: {checksum}")


if __name__ == "__main__":
    main()
