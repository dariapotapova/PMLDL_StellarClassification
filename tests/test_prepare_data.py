"""Preprocessing tests for the sentinel-cleaning helper and split utilities.

References
----------
pandas ``isna`` documentation:
https://pandas.pydata.org/docs/reference/api/pandas.isna.html

pandas ``NA`` scalar documentation:
https://pandas.pydata.org/docs/reference/api/pandas.NA.html
"""

import numpy as np
import pandas as pd
import pytest

from src.prepare_data import SENTINEL_COLUMNS, SENTINEL_VALUE, clean_sentinel_values


def test_clean_sentinel_values_replaces_raw_band_sentinels() -> None:
    frame = pd.DataFrame(
        {
            "u": [SENTINEL_VALUE, 19.1, 18.5],
            "g": [17.2, SENTINEL_VALUE, 17.8],
            "z": [13.4, 13.9, SENTINEL_VALUE],
            "alpha": [0.1, 0.2, 0.3],
        }
    )

    cleaned = clean_sentinel_values(frame)

    assert pd.isna(cleaned.loc[0, "u"])
    assert pd.isna(cleaned.loc[1, "g"])
    assert pd.isna(cleaned.loc[2, "z"])
    assert cleaned.loc[1, "u"] == 19.1
    assert cleaned.loc[0, "g"] == 17.2
    assert list(cleaned["alpha"]) == [0.1, 0.2, 0.3]


def test_clean_sentinel_values_replaces_derived_colour_sentinels() -> None:
    frame = pd.DataFrame(
        {
            "u_g": [SENTINEL_VALUE, 1.4, 0.8],
            "g_r": [0.6, SENTINEL_VALUE, 0.9],
            "i_z": [0.1, 0.2, SENTINEL_VALUE],
            "u_r": [SENTINEL_VALUE, 2.1, 1.7],
            "r_i": [0.0, 0.0, -0.5],
        }
    )

    cleaned = clean_sentinel_values(frame)

    assert pd.isna(cleaned.loc[0, "u_g"])
    assert pd.isna(cleaned.loc[1, "g_r"])
    assert pd.isna(cleaned.loc[2, "i_z"])
    assert pd.isna(cleaned.loc[0, "u_r"])
    assert list(cleaned["r_i"]) == [0.0, 0.0, -0.5]


def test_clean_sentinel_values_preserves_nonlisted_columns_even_with_sentinel() -> None:
    frame = pd.DataFrame(
        {
            "u": [SENTINEL_VALUE],
            "redshift": [SENTINEL_VALUE],
            "class_encoded": [SENTINEL_VALUE],
            "delta": [SENTINEL_VALUE],
        }
    )

    cleaned = clean_sentinel_values(frame)

    assert pd.isna(cleaned.loc[0, "u"])
    assert cleaned.loc[0, "redshift"] == SENTINEL_VALUE
    assert cleaned.loc[0, "class_encoded"] == SENTINEL_VALUE
    assert cleaned.loc[0, "delta"] == SENTINEL_VALUE


def test_clean_sentinel_values_returns_copy_original_frame_unchanged() -> None:
    original = pd.DataFrame(
        {"u": [SENTINEL_VALUE, 19.0], "g_r": [0.5, SENTINEL_VALUE]}
    )

    cleaned = clean_sentinel_values(original)

    assert original.loc[0, "u"] == SENTINEL_VALUE
    assert original.loc[1, "g_r"] == SENTINEL_VALUE
    assert pd.isna(cleaned.loc[0, "u"])
    assert pd.isna(cleaned.loc[1, "g_r"])
    assert cleaned is not original


def test_sentinel_columns_list_covers_expected_nine_features() -> None:
    assert "u" in SENTINEL_COLUMNS
    assert "g" in SENTINEL_COLUMNS
    assert "z" in SENTINEL_COLUMNS
    for derived in ("u_g", "g_r", "r_i", "i_z", "u_r", "g_i"):
        assert derived in SENTINEL_COLUMNS
    assert len(SENTINEL_COLUMNS) == 9
