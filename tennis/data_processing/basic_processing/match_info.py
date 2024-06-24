"""Functionality to clean and validate the match_info.csv file."""

import pandas as pd


def filter_match_info_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the match info CSV to exclude Davis Cup tournaments."""
    df["tourney_name"] = df["tourney_name"].str.lower()
    df = df[~df["tourney_name"].str.contains("davis cup")]
    return df
