"""Functions for using Basic Markov Model. We get player mean serve win percentage for both players and use it to predict the winner of a match."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss

import mlflow.sklearn

from tennis.model_usage.config import DEFAULT_RANDOM_SEED
from tennis.models.markov_model import (
    get_player_1_match_winning_probability,
    TennisParameters,
)

# Load the final DataFrame
final_df = pd.read_csv("./data/final_data.csv")

# filter for games_played_by_player > 20 and games_played_by_opponent > 20
final_df = final_df[
    (final_df["games_played_by_player"] > 20)
    & (final_df["games_played_by_opponent"] > 20)
]

# Define features and target
X = final_df[["player_id", "opponent_id"]]
y = final_df["win"]

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", DEFAULT_RANDOM_SEED)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=DEFAULT_RANDOM_SEED
    )

    # Filter training data for fitting the mixed effects model
    test_data = final_df.loc[X_test.index]

    # get player and opponent avg serve win percentage from column historic_total_serve_win_pct
    player_avg_serve_win_pct = test_data["historic_player_total_serve_win_pct"]
    opponent_avg_serve_win_pct = test_data["historic_opponent_total_serve_win_pct"]
    n_sets = test_data["best_of"]

    # use all above to predict winning prob

    y_pred_proba = []
    for i in range(len(player_avg_serve_win_pct)):
        y_pred_proba.append(
            get_player_1_match_winning_probability(
                TennisParameters(
                    player_avg_serve_win_pct.iloc[i], opponent_avg_serve_win_pct.iloc[i]
                ),
                int(n_sets.iloc[i]),
            )
        )

    # Evaluate the model
    log_loss_value = log_loss(y_test, y_pred_proba)
    brier_score_value = brier_score_loss(y_test, y_pred_proba)
    # accuracy_value = accuracy_score(y_test, y_pred_proba > 0.5)

    # Log metrics
    mlflow.log_metric("log_loss", log_loss_value)
    mlflow.log_metric("brier_score", brier_score_value)
    # mlflow.log_metric("accuracy", accuracy_value)

    print(f"Log Loss: {log_loss_value}")
    print(f"Brier Score: {brier_score_value}")
