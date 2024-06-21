"""Functionality to clean and validate the match_info.csv file."""

from datetime import datetime
from enum import Enum

import pandas as pd
from pydantic import BaseModel, constr, field_validator, PositiveInt
from fuzzywuzzy import fuzz


class TournamentSurface(Enum):
    """Enum for the surface of the tournament."""

    CARPET = "Carpet"
    CLAY = "Clay"
    GRASS = "Grass"
    HARD = "Hard"


class TournamentLevel(Enum):
    """Enum for the level of the tournament. Unsure what the values are."""

    A = "A"
    C = "C"
    D = "D"
    F = "F"
    G = "G"
    M = "M"


class TournamentRound(Enum):
    """Enum for the round of the tournament."""

    R128 = "R128"
    R64 = "R64"
    R32 = "R32"
    R16 = "R16"
    QF = "QF"
    SF = "SF"
    F = "F"
    RR = "RR"  # Round Robin
    BR = "BR"  # Bronze Medal Match ?


class MatchInfo(BaseModel):
    match_id: int
    tourney_id: int
    tourney_name: constr(to_lower=True)
    tourney_date: datetime
    tourney_level: TournamentLevel
    surface: TournamentSurface
    match_num: PositiveInt
    best_of: PositiveInt
    round: TournamentRound

    @field_validator("best_of")
    @classmethod
    def check_best_of(cls, value):
        """Check that the best of value is either 3 or 5."""
        if value not in [3, 5]:
            msg = f"Best of must be 3 or 5, not {value}"
            raise ValueError(msg)
        return value


class MatchInfoList(BaseModel):
    data: list[MatchInfo]




# Load the data
df = pd.read_csv("./data/match_info.csv")

# Check for duplicate match_id
duplicates = df[df.duplicated(['match_id'])]
if not duplicates.empty:
    print(f"Duplicates found: {duplicates}")

# Create MatchInfo instances
match_list = [MatchInfo(**row._asdict()) for row in df.itertuples(index=False)]

match_info_list = MatchInfoList(data=match_list)

# Check string similarity for tournament names
for index, row in df.iterrows():
    for other_index, other_row in df.iterrows():
        if (
                index != other_index
                and fuzz.ratio(row["tourney_name"], other_row["tourney_name"]) > 80
        ):
            print(f"Row {index} and Row {other_index} have similar tournament names")

# Save the cleaned data
df.to_csv("data/cleaned_match_info.csv", index=False)
