"""Pydantic models for the player info data."""

from enum import Enum

import pandas as pd
from pydantic import BaseModel, constr, field_validator, PositiveInt


class PlayerHand(Enum):
    """Enum for the hand of the player."""

    LEFT = "L"
    RIGHT = "R"


# match_id,winner_name,loser_name,winner_age,loser_age,winner_rank,loser_rank,winner_rank_points,loser_rank_points,winner_seed,loser_seed,winner_ioc,loser_ioc,winner_hand,loser_hand
# 0,Antony Dupuis,Andrew Ilie,27.1813826146,24.0355920602,113,50,351.0,762.0,,1,FRA,AUS,R,R
# 1,Fernando Gonzalez,Cecil Mamiit,19.7563312799,23.8439425051,352,139,76.0,280.0,,,CHI,PHI,R,R
# 2,Paradorn Srichaphan,Sebastien Lareau,20.8815879535,27.0116358658,103,133,380.0,293.0,,,THA,CAN,R,R
class PlayerInfo(BaseModel):
    match_id: int
    winner_name: constr(to_lower=True)  # type: ignore
    loser_name: constr(to_lower=True)  # type: ignore
    winner_age: float
    loser_age: float
    winner_rank: PositiveInt | None
    loser_rank: PositiveInt | None
    winner_rank_points: float
    loser_rank_points: float
    winner_seed: PositiveInt | None
    loser_seed: PositiveInt | None
    winner_ioc: constr(to_lower=True)  # type: ignore
    loser_ioc: constr(to_lower=True)  # type: ignore
    winner_hand: PlayerHand
    loser_hand: PlayerHand

    @field_validator("winner_hand")
    @classmethod
    def check_winner_hand(cls, value):
        """Check that the winner hand is either left or right."""
        if value not in [PlayerHand.LEFT, PlayerHand.RIGHT]:
            msg = f"Winner hand must be 'L' or 'R', not {value}"
            raise ValueError(msg)
        return value

    @field_validator("loser_hand")
    @classmethod
    def check_loser_hand(cls, value):
        """Check that the loser hand is either left or right."""
        if value not in [PlayerHand.LEFT, PlayerHand.RIGHT]:
            msg = f"Loser hand must be 'L' or 'R', not {value}"
            raise ValueError(msg)
        return value


# # Load the data
df = pd.read_csv("./data/player_info.csv")
# convert to pydnatic model using as_dict
player_info_list = [PlayerInfo(**row._asdict()) for row in df.itertuples(index=False)]
