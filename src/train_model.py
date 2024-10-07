import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import mlflow
import mlflow.sklearn

'''

Copy and paste into terminal

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root mlruns \
  --host 0.0.0.0 \
  --port 5001

'''

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")

def load_preprocessed_data():
    """
    Load the preprocessed training and validation data.

    Returns:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training target.
    - X_val (DataFrame): Validation features.
    - y_val (Series): Validation target.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the data directory
    data_dir = os.path.join(script_dir, '..', 'data')

    # Load training data
    train_data_path = os.path.join(data_dir, 'train_preprocessed.csv')
    if not os.path.isfile(train_data_path):
        raise FileNotFoundError(f"Training data not found at {train_data_path}")
    train_data = pd.read_csv(train_data_path)

    # Load validation data
    val_data_path = os.path.join(data_dir, 'val_preprocessed.csv')
    if not os.path.isfile(val_data_path):
        raise FileNotFoundError(f"Validation data not found at {val_data_path}")
    val_data = pd.read_csv(val_data_path)

    # Separate features and target
    X_train = train_data.drop('RUL', axis=1)
    y_train = train_data['RUL']
    X_val = val_data.drop('RUL', axis=1)
    y_val = val_data['RUL']

    return X_train, y_train, X_val, y_val

def train_and_evaluate_model(X_train, y_train, X_val, y_val):
    """
    Train the machine learning model and evaluate its performance.

    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training target.
    - X_val (DataFrame): Validation features.
    - y_val (Series): Validation target.

    Returns:
    - model (sklearn model): Trained model.
    - metrics (dict): Dictionary containing evaluation metrics.
    """
    # Set the experiment name in MLflow
    mlflow.set_experiment("Predictive Maintenance Model Training")

    # Start an MLflow run
    with mlflow.start_run():
        # Define model parameters
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'random_state': 42,
            'n_jobs': -1
        }

        # Log parameters to MLflow
        mlflow.log_params(params)

        # Initialize the model
        model = RandomForestRegressor(**params)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_pred = model.predict(X_val)

        # Evaluate the model
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = model.score(X_val, y_val)

        # Log metrics to MLflow
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2', r2)

        print(f"Validation RMSE: {rmse:.2f}")
        print(f"Validation MAE: {mae:.2f}")
        print(f"Validation R2 Score: {r2:.2f}")

        # Log the model to MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Return the trained model and metrics
        metrics = {'rmse': rmse, 'mae': mae, 'r2': r2}
        return model, metrics

def save_model(model):
    """
    Save the trained model to the models directory.

    Parameters:
    - model (sklearn model): Trained model.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the models directory
    models_dir = os.path.join(script_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(models_dir, 'rf_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def main():
    # Load the data
    X_train, y_train, X_val, y_val = load_preprocessed_data()

    # Train and evaluate the model
    model, metrics = train_and_evaluate_model(X_train, y_train, X_val, y_val)

    # Save the model
    save_model(model)

if __name__ == '__main__':
    main()
