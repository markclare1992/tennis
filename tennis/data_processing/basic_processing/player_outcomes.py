import pandas as pd
import numpy as np


def pivot_player_outcome_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the player outcome stats DataFrame from long to wide format.

    Parameters:
    - df (pd.DataFrame): The player outcome stats DataFrame.

    Returns:
    - pd.DataFrame: The pivoted DataFrame.
    """
    pivoted_df = df.pivot(
        index=["match_id", "player_name"], columns="stat", values="stat_value"
    ).reset_index()

    # Rename columns to add player_ prefix
    pivoted_df = pivoted_df.rename(
        columns=lambda x: "player_" + x if x not in ["match_id", "player_name"] else x
    )

    return pivoted_df


def safe_divide(numerator, denominator):
    """Safely divide two arrays, handling division by zero."""
    return np.where(denominator == 0, 0, numerator / denominator)


def add_transformed_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add transformed variables to the player stats DataFrame."""
    df["player_second_serves"] = df["player_svpt"] - df["player_firstin"]
    df["player_total_serve_points_won"] = df["player_firstwon"] + df["player_secondwon"]
    df["player_total_serve_points_lost"] = (
        df["player_svpt"] - df["player_total_serve_points_won"]
    )
    df["player_first_serve_pct"] = safe_divide(df["player_firstin"], df["player_svpt"])
    df["player_first_serve_win_pct"] = safe_divide(
        df["player_firstwon"], df["player_firstin"]
    )
    df["player_second_serve_win_pct"] = safe_divide(
        df["player_secondwon"], df["player_second_serves"]
    )
    df["player_total_serve_win_pct"] = safe_divide(
        df["player_total_serve_points_won"], df["player_svpt"]
    )
    df["player_df_pct"] = safe_divide(df["player_df"], df["player_svpt"])
    df["player_bpsaved_pct"] = safe_divide(df["player_bpsaved"], df["player_bpfaced"])
    df["player_aces_per_game"] = safe_divide(df["player_ace"], df["player_svgms"])
    df["player_ace_pct"] = safe_divide(df["player_ace"], df["player_svpt"])
    df["player_df_per_game"] = safe_divide(df["player_df"], df["player_svgms"])
    df["player_serve_points_won_per_game"] = safe_divide(
        df["player_total_serve_points_won"], df["player_svgms"]
    )
    df["player_serve_points_lost_per_game"] = safe_divide(
        df["player_total_serve_points_lost"], df["player_svgms"]
    )
    return df


def merge_with_opponent_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the player stats DataFrame with itself to include opponent stats.

    Parameters:
    - df (pd.DataFrame): The player stats DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with both player and opponent stats.
    """
    merged_df = pd.merge(
        df,
        df,
        how="inner",
        left_on=["match_id", "player_name"],
        right_on=["match_id", "player_name"],
        suffixes=("", "_opponent"),
    )

    # Rename opponent columns
    merged_df = merged_df.rename(
        columns=lambda x: x.replace("_opponent", "").replace("player_", "opponent_")
        if "_opponent" in x
        else x
    )

    return merged_df
