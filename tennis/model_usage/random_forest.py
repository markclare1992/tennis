import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

import mlflow.sklearn

from tennis.model_usage.config import DEFAULT_RANDOM_SEED

# Load the final DataFrame
final_df = pd.read_csv("./data/final_data.csv")

# Prepare the features and target variable
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

    # Train the random forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # Evaluate the model
    log_loss_value = log_loss(y_test, y_pred_proba)
    brier_score_value = brier_score_loss(y_test, y_pred_proba[:, 1])
    accuracy_value = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("log_loss", log_loss_value)
    mlflow.log_metric("brier_score", brier_score_value)
    mlflow.log_metric("accuracy", accuracy_value)

    print(f"Log Loss: {log_loss_value}")
    print(f"Brier Score: {brier_score_value}")
    print(f"Accuracy: {accuracy_value}")
