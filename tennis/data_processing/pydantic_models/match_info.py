"""Pydantic models for the match info data."""
from datetime import datetime
from enum import Enum

import pandas as pd
from pydantic import BaseModel, constr, field_validator, PositiveInt

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
    tourney_name: constr(to_lower=True)  # type: ignore
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