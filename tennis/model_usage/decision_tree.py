import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier

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
    # Train the decision tree model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(model, "decision_tree_model")

    # Predict probabilities
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    print(f"Accuracy: {accuracy}")
