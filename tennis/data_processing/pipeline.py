#load match info csv, convert tourney_name to lower, remove davis cup, save to csv
# Load the data
import pandas as pd

from tennis.data_processing.match_info import filter_match_info_csv
from tennis.data_processing.match_scores import process_match_scores
from tennis.data_processing.player_outcomes import add_transformed_variables, \
    pivot_player_outcome_stats

match_info_df = pd.read_csv("./data/match_info.csv")
filtered_df = filter_match_info_csv(match_info_df)
#can possibly use pydantic for validation in future added in

#load match outcome stats, parse scores, remove any uncompleted matches, save to csv
# extension, incompleted matches still provide some value, so dont throw them yet.
match_outcome_df = pd.read_csv("./data/match_outcome_stats.csv")
match_outcome_df = process_match_scores(match_outcome_df)

# player info
player_info = pd.read_csv("./data/player_info.csv")

# player outcomes
player_outcomes = pd.read_csv("./data/player_outcome_stats.csv")
player_outcomes_pivoted = pivot_player_outcome_stats(player_outcomes)
player_outcomes_pivoted = add_transformed_variables(player_outcomes_pivoted)

#merge match info and match outcome stats
merged_df = pd.merge(
    filtered_df, match_outcome_df, how="inner", on="match_id"
)
