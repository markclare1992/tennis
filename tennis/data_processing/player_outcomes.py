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
    return df.pivot(
        index=["match_id", "player_name"], columns="stat", values="stat_value"
    ).reset_index()


# Safely divide to avoid division by zero
def safe_divide(numerator, denominator):
    return np.where(denominator == 0, 0, numerator / denominator)


def add_transformed_variables(df:pd.DataFrame) -> pd.DataFrame:
    df['second_serves'] = df['svpt'] - df['firstin']
    df['total_serve_points_won'] = df['firstwon'] + df['secondwon']
    df['total_serve_points_lost'] = df['svpt'] - df['total_serve_points_won']
    df['first_serve_pct'] = safe_divide(df['firstin'], df['svpt'])
    df['first_serve_win_pct'] = safe_divide(df['firstwon'], df['firstin'])
    df['second_serve_win_pct'] = safe_divide(df['secondwon'], df['second_serves'])
    df['total_serve_win_pct'] = safe_divide(df['total_serve_points_won'], df['svpt'])
    df['df_pct'] = safe_divide(df['df'], df['svpt'])
    df['bpsaved_pct'] = safe_divide(df['bpsaved'], df['bpfaced'])
    df['aces_per_game'] = safe_divide(df['ace'], df['svgms'])
    df['ace_pct'] = safe_divide(df['ace'], df['svpt'])
    df['df_per_game'] = safe_divide(df['df'], df['svgms'])
    df['serve_points_won_per_game'] = safe_divide(df['total_serve_points_won'], df['svgms'])
    df['serve_points_lost_per_game'] = safe_divide(df['total_serve_points_lost'], df['svgms'])
    return df

