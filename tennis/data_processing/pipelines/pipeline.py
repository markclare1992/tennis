"""Script to run the entire data processing pipeline."""

import pandas as pd

from tennis.data_processing.basic_processing.match_info import filter_match_info_csv
from tennis.data_processing.basic_processing.match_scores import process_match_scores
from tennis.data_processing.basic_processing.player_info import (
    player_info_wide_to_long,
    create_player_ids_from_dataframe,
)
from tennis.data_processing.basic_processing.player_outcomes import (
    add_transformed_variables,
    pivot_player_outcome_stats,
    merge_with_opponent_stats,
)
from tennis.data_processing.historical_features.historic_featureset import (
    calculate_historic_features,
)


def run_pipeline():
    """Run the entire data processing pipeline."""
    # Load and process match information
    match_info_df = pd.read_csv("./data/match_info.csv")
    filtered_match_info_df = filter_match_info_csv(match_info_df)

    # Load and process match outcomes
    match_outcome_df = pd.read_csv("./data/match_outcome_stats.csv")
    processed_match_outcome_df = process_match_scores(match_outcome_df)

    # Load and process player information
    player_info_df = pd.read_csv("./data/player_info.csv")

    # Load player Ids
    player_ids = create_player_ids_from_dataframe(player_info_df)
    player_ids.to_csv("./data/player_ids.csv", index=False)

    player_info_long_df = player_info_wide_to_long(player_info_df)

    # Load, pivot, and process player outcomes
    player_outcomes_df = pd.read_csv("./data/player_outcome_stats.csv")
    player_outcomes_pivoted_df = pivot_player_outcome_stats(player_outcomes_df)
    player_outcomes_transformed_df = add_transformed_variables(
        player_outcomes_pivoted_df
    )
    player_outcomes_with_opponent_df = merge_with_opponent_stats(
        player_outcomes_transformed_df
    )

    # Merge match info and match outcomes
    merged_match_info_outcome_df = pd.merge(
        filtered_match_info_df, processed_match_outcome_df, how="inner", on="match_id"
    )

    # Merge player info and player outcomes
    merged_player_info_outcome_df = pd.merge(
        player_info_long_df,
        player_outcomes_with_opponent_df,
        left_on=["match_id", "player_name"],
        right_on=["match_id", "player_name"],
    )

    # Merge all data
    merged_all_df = pd.merge(
        merged_match_info_outcome_df,
        merged_player_info_outcome_df,
        how="inner",
        on="match_id",
    )

    # Calculate historic features
    historic_features_df = calculate_historic_features(merged_all_df)

    # Save the final DataFrame
    historic_features_df.to_csv("./data/historic_features.csv", index=False)

    # Merge the historic features with the final DataFrame
    final_df = pd.merge(
        merged_all_df,
        historic_features_df,
        how="inner",
        on=["match_id", "player_name", "opponent_name"],
    )

    # Merge the player ids with the final DataFrame to get player ids and opponent ids
    final_df = pd.merge(final_df, player_ids, how="inner", on=["player_name"])
    player_ids = player_ids.rename(
        columns={"player_id": "opponent_id", "player_name": "opponent_name"}
    )
    final_df = pd.merge(final_df, player_ids, how="inner", on=["opponent_name"])

    # Save the final DataFrame
    final_df.to_csv("./data/final_data.csv", index=False)


if __name__ == "__main__":
    run_pipeline()
