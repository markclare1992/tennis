"""Functionality to clean and validate the match_info.csv file."""

from datetime import datetime
from enum import Enum

import pandas as pd
from pydantic import BaseModel, constr, field_validator, PositiveInt

def filter_match_info_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the match info CSV to exclude Davis Cup tournaments."""
    df["tourney_name"] = df["tourney_name"].str.lower()
    df = df[~df["tourney_name"].str.contains("davis cup")]
    return df

# # Load the data
# df = pd.read_csv("./data/match_info.csv")
# filtered_df = filter_match_info_csv(df)
#
#
#
# # Save the cleaned data
# df.to_csv("data/cleaned_match_info.csv", index=False)
