import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

import mlflow.sklearn

from tennis.model_usage.config import DEFAULT_RANDOM_SEED

# Load the final DataFrame
final_df = pd.read_csv("./data/final_data.csv")

# Prepare the features and target variable
X = final_df[["player_id", "opponent_id"]]
y = final_df["win"]

# Start an MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", DEFAULT_RANDOM_SEED)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=DEFAULT_RANDOM_SEED
    )

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(model, "logistic_regression_model")

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)

    # Evaluate the model
    log_loss_value = log_loss(y_test, y_pred_proba)
    brier_score_value = brier_score_loss(y_test, y_pred_proba[:, 1])

    # Log metrics
    mlflow.log_metric("log_loss", log_loss_value)
    mlflow.log_metric("brier_score", brier_score_value)

    print(f"Log Loss: {log_loss_value}")
    print(f"Brier Score: {brier_score_value}")
