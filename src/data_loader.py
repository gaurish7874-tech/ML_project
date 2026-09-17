"""
Data Loader Module for NASA C-MAPSS Turbofan Degradation Simulation (FD001).

This module provides robust, clean utilities to ingest raw space-separated text files,
remove empty/accidental spacing columns, assign standardized column names,
and validate data schemas without modifying underlying raw values.
"""

from pathlib import Path
from typing import Tuple, Union, Optional
import pandas as pd


# Standardized Column Name Definitions for NASA C-MAPSS Datasets
IDENTIFIER_COLUMNS = ["unit_number", "time_cycles"]
SETTING_COLUMNS = [
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
TELEMETRY_COLUMNS = IDENTIFIER_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS
RUL_COLUMNS = ["RUL"]


def _clean_and_assign_columns(
    df: pd.DataFrame, expected_columns: list, dataset_name: str
) -> pd.DataFrame:
    """
    Cleans DataFrame from accidental spacing columns and assigns expected column names.

    Parameters
    ----------
    df : pd.DataFrame
        Raw loaded DataFrame.
    expected_columns : list
        List of target column names.
    dataset_name : str
        Name of the dataset for logging and validation error messages.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with assigned column names and unchanged raw values.

    Raises
    ------
    ValueError
        If the number of non-empty columns does not match expected_columns length.
    """
    # Drop columns that are completely empty / all NaN (often caused by trailing spaces)
    df = df.dropna(how="all", axis=1).copy()

    # Schema length validation
    if df.shape[1] != len(expected_columns):
        raise ValueError(
            f"Schema mismatch for {dataset_name}: Expected {len(expected_columns)} columns, "
            f"but found {df.shape[1]} non-empty columns. "
            f"Columns received: {list(df.columns)}"
        )

    # Assign standardized names
    df.columns = expected_columns
    return df


def load_train_data(
    data_dir: Union[str, Path] = "data/raw",
    filename: str = "train_FD001.txt",
) -> pd.DataFrame:
    """
    Loads training run-to-failure trajectories for the FD001 dataset.

    Parameters
    ----------
    data_dir : Union[str, Path], default='data/raw'
        Path to the directory containing raw data files.
    filename : str, default='train_FD001.txt'
        Name of the training dataset file.

    Returns
    -------
    pd.DataFrame
        Raw training data with 26 standardized columns.
    """
    filepath = Path(data_dir) / filename
    if not filepath.is_file():
        raise FileNotFoundError(f"Training data file not found at: {filepath.resolve()}")

    df_raw = pd.read_csv(filepath, sep=r"\s+", header=None, engine="python")
    df_clean = _clean_and_assign_columns(df_raw, TELEMETRY_COLUMNS, "train_FD001")
    return df_clean


def load_test_data(
    data_dir: Union[str, Path] = "data/raw",
    filename: str = "test_FD001.txt",
) -> pd.DataFrame:
    """
    Loads censored test trajectories for the FD001 dataset.

    Parameters
    ----------
    data_dir : Union[str, Path], default='data/raw'
        Path to the directory containing raw data files.
    filename : str, default='test_FD001.txt'
        Name of the test dataset file.

    Returns
    -------
    pd.DataFrame
        Raw test data with 26 standardized columns.
    """
    filepath = Path(data_dir) / filename
    if not filepath.is_file():
        raise FileNotFoundError(f"Test data file not found at: {filepath.resolve()}")

    df_raw = pd.read_csv(filepath, sep=r"\s+", header=None, engine="python")
    df_clean = _clean_and_assign_columns(df_raw, TELEMETRY_COLUMNS, "test_FD001")
    return df_clean


def load_rul_data(
    data_dir: Union[str, Path] = "data/raw",
    filename: str = "RUL_FD001.txt",
) -> pd.DataFrame:
    """
    Loads ground truth Remaining Useful Life (RUL) values for the test engines.

    Parameters
    ----------
    data_dir : Union[str, Path], default='data/raw'
        Path to the directory containing raw data files.
    filename : str, default='RUL_FD001.txt'
        Name of the ground truth RUL file.

    Returns
    -------
    pd.DataFrame
        Ground truth RUL with standardized column ['RUL'].
    """
    filepath = Path(data_dir) / filename
    if not filepath.is_file():
        raise FileNotFoundError(f"Ground truth RUL file not found at: {filepath.resolve()}")

    df_raw = pd.read_csv(filepath, sep=r"\s+", header=None, engine="python")
    df_clean = _clean_and_assign_columns(df_raw, RUL_COLUMNS, "RUL_FD001")
    return df_clean


def load_cmapss_data(
    data_dir: Union[str, Path] = "data/raw",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper to load all three NASA C-MAPSS FD001 datasets simultaneously.

    Parameters
    ----------
    data_dir : Union[str, Path], default='data/raw'
        Path to the directory containing raw data files.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tuple containing (train_df, test_df, rul_df).
    """
    train_df = load_train_data(data_dir=data_dir)
    test_df = load_test_data(data_dir=data_dir)
    rul_df = load_rul_data(data_dir=data_dir)
    return train_df, test_df, rul_df


if __name__ == "__main__":
    print("Testing data_loader.py...")
    train, test, rul = load_cmapss_data()
    print(f"Train shape: {train.shape}, Unit numbers: {train['unit_number'].nunique()}")
    print(f"Test shape:  {test.shape},  Unit numbers: {test['unit_number'].nunique()}")
    print(f"RUL shape:   {rul.shape},   Total test engines: {len(rul)}")
    print("Columns in train:")
    print(list(train.columns))
    print("All assertions and validations passed successfully!")
