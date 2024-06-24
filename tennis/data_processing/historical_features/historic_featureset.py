"""Functionality for calculating features at a given date."""

import pandas as pd


def calculate_cumulative_mean(df, group_cols, feature):
    """
    Calculate the cumulative mean for a feature grouped by specific columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        group_cols (list): List of columns to group by.
        feature (str): The feature column to calculate the cumulative mean for.

    Returns:
        pd.Series: The cumulative mean series.
    """
    return (
        df.groupby(group_cols)[feature]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )


def calculate_games_played(df, group_cols):
    """
    Calculate the cumulative count of games played by each player.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        group_cols (list): List of columns to group by.

    Returns:
        pd.Series: The cumulative count series.
    """
    return df.groupby(group_cols).cumcount()


def calculate_historic_features(final_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate historic features for players and opponents.

    Parameters:
    - final_dataframe: The final DataFrame containing all the data.

    Returns:
    - pd.DataFrame: The DataFrame with calculated historic features.
    """
    features = [
        "ace_pct",
        "aces_per_game",
        "bpsaved_pct",
        "df_pct",
        "df_per_game",
        "first_serve_pct",
        "first_serve_win_pct",
        "second_serve_win_pct",
        "serve_points_lost_per_game",
        "serve_points_won_per_game",
        "total_serve_win_pct",
    ]

    # Sort the DataFrame by tourney_date and match_num to ensure proper ordering
    final_dataframe = final_dataframe.sort_values(by=["tourney_date", "match_num"])

    historic_features_df = pd.DataFrame()
    historic_features_df["match_id"] = final_dataframe["match_id"]
    historic_features_df["player_name"] = final_dataframe["player_name"]
    historic_features_df["opponent_name"] = final_dataframe["opponent_name"]

    # Calculate games played for players and opponents
    historic_features_df["games_played_by_player"] = calculate_games_played(
        final_dataframe, ["player_name"]
    )
    historic_features_df["games_played_by_opponent"] = calculate_games_played(
        final_dataframe, ["opponent_name"]
    )

    for feature in features:
        # Calculate historic features for players
        historic_features_df[f"historic_player_{feature}"] = final_dataframe.groupby(
            ["player_name", "tourney_date"]
        )[f"player_{feature}"].transform(lambda x: x.expanding().mean().shift())
        # Calculate historic features for opponents
        historic_features_df[f"historic_opponent_{feature}"] = final_dataframe.groupby(
            ["opponent_name", "tourney_date"]
        )[f"opponent_{feature}"].transform(lambda x: x.expanding().mean().shift())

    historic_features_df.fillna(
        0, inplace=True
    )  # Handle NaN values for the first matches

    return historic_features_df
