import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

from tennis.model_usage.decision_tree import train_decision_tree
from tennis.model_usage.logistic_regression import train_logistic_regression

# Load the final DataFrame
final_df = pd.read_csv("./data/final_data.csv")

# Prepare the features and target variable
X = final_df[["player_id", "opponent_id"]]
y = final_df["win"]

# Split the data (ensure the same random state is used for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Log the parameters
mlflow.log_param("test_size", 0.2)
mlflow.log_param("random_state", 42)

# Train and log the logistic regression model
train_logistic_regression(X_train, X_test, y_train, y_test)
train_decision_tree(X_train, X_test, y_train, y_test)
