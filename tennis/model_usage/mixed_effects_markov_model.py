"""Functions for fitting a mixed effects Markov model to tennis data."""

import statsmodels.formula.api as smf
import pandas as pd
from sklearn.model_selection import train_test_split

import mlflow.sklearn

from tennis.model_usage.config import DEFAULT_RANDOM_SEED


# Load the final DataFrame
final_df = pd.read_csv("./data/final_data.csv")


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
    train_data = final_df.loc[X_train.index]

    # Fit the mixed effects model
    player_model = smf.mixedlm(
        "player_total_serve_win_pct ~ 1",
        train_data,
        groups=train_data["player_id"],
        re_formula="~opponent_id",
    )
    player_model_fit = player_model.fit()

    # Log the mixed effects model
    mlflow.log_param("model_type", "mixed_effects")
    with open("mixed_effects_model_summary.txt", "w") as f:
        f.write(player_model_fit.summary().as_text())
    mlflow.log_artifact("mixed_effects_model_summary.txt")

    test_data = final_df.loc[X_test.index]
    ...
