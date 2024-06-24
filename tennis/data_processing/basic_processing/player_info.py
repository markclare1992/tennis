import pandas as pd


def player_info_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the player info DataFrame from wide to long format.

    Parameters:
        - df (pd.DataFrame): The player info DataFrame.

    Returns:
        - pd.DataFrame: The long-format DataFrame.
    """
    transformed_data = []

    for _, row in df.iterrows():
        match_id = row["match_id"]

        winner_data = {
            "match_id": match_id,
            "player_name": row["winner_name"],
            "opponent_name": row["loser_name"],
            "player_age": row["winner_age"],
            "opponent_age": row["loser_age"],
            "player_rank": row["winner_rank"],
            "opponent_rank": row["loser_rank"],
            "player_rank_points": row["winner_rank_points"],
            "opponent_rank_points": row["loser_rank_points"],
            "player_seed": row["winner_seed"],
            "opponent_seed": row["loser_seed"],
            "player_ioc": row["winner_ioc"],
            "opponent_ioc": row["loser_ioc"],
            "player_hand": row["winner_hand"],
            "opponent_hand": row["loser_hand"],
            "outcome": "win",
            "win": 1,
        }

        loser_data = {
            "match_id": match_id,
            "player_name": row["loser_name"],
            "opponent_name": row["winner_name"],
            "player_age": row["loser_age"],
            "opponent_age": row["winner_age"],
            "player_rank": row["loser_rank"],
            "opponent_rank": row["winner_rank"],
            "player_rank_points": row["loser_rank_points"],
            "opponent_rank_points": row["winner_rank_points"],
            "player_seed": row["loser_seed"],
            "opponent_seed": row["winner_seed"],
            "player_ioc": row["loser_ioc"],
            "opponent_ioc": row["winner_ioc"],
            "player_hand": row["loser_hand"],
            "opponent_hand": row["winner_hand"],
            "outcome": "lose",
            "win": 0,
        }

        transformed_data.append(winner_data)
        transformed_data.append(loser_data)

    return pd.DataFrame(transformed_data)


def create_player_ids_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame of unique player IDs from a DataFrame of player info.

    Parameters:
    - df (pd.DataFrame): The player info DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame of unique player IDs.

    Note:
        Intended to only run on player_info dataframe due to hardcoded column names.
    """
    player_ids = (
        pd.concat([df["winner_name"], df["loser_name"]])
        .drop_duplicates()
        .reset_index(drop=True)
        .reset_index()
    )
    player_ids.columns = ["player_id", "player_name"]
    return player_ids
