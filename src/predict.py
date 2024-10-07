import os
import argparse
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
import logging
from src.api.utils import clean_input_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)

def load_model_joblib(model_path):
    """
    Load the trained model from a joblib file.
    """
    logging.info(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    try:
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load the model: {e}")
        raise RuntimeError(f"Failed to load the model: {e}")
    return model
def load_model_joblib(model_path):
    """
    Load the trained model from a joblib file.

    Parameters:
    - model_path (str): Path to the joblib model file.

    Returns:
    - model: Loaded machine learning model.
    """
    print(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load the model: {e}")
    return model

def load_model_mlflow(model_name, run_id=None):
    """
    Load the trained model from MLflow.

    Parameters:
    - model_name (str): Name of the registered MLflow model.
    - run_id (str, optional): Specific run ID to load the model from.

    Returns:
    - model: Loaded machine learning model.
    """
    # Set the MLflow tracking URI (ensure this matches your server)
    mlflow.set_tracking_uri("http://127.0.0.1:5001")

    if run_id:
        # Load model from a specific run
        model_uri = f"runs:/{run_id}/model"
    else:
        # Load the latest model from the model registry
        model_uri = f"models:/{model_name}/latest"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded successfully from MLflow: {model_uri}")
    except Exception as e:
        raise RuntimeError(f"Failed to load the model from MLflow: {e}")
    return model

def load_and_preprocess_data(input_path, scaler_path=None):
    """
    Load and preprocess new input data.

    Parameters:
    - input_path (str): Path to the new input CSV file.
    - scaler_path (str, optional): Path to the scaler file if scaling is needed.

    Returns:
    - X_new (DataFrame): Preprocessed features ready for prediction.
    """
    print(f"Loading input data from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data file not found at {input_path}")

    # Load data
    data = pd.read_csv(input_path)

    # Drop 'RUL' column if it exists
    if 'RUL' in data.columns:
        print("Warning: 'RUL' column found in input data. Dropping it for prediction.")
        data = data.drop(columns=['RUL'])

    # Handle missing values (if any)
    data = data.dropna()
    print(f"Data shape after dropping missing values: {data.shape}")

    # Feature Engineering (if required)
    # Example: data['feature_new'] = data['feature1'] * data['feature2']

    # Feature Scaling (if it was done during training)
    if scaler_path:
        print(f"Loading scaler from: {scaler_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        scaler = joblib.load(scaler_path)
        print("Applying scaler to input data.")
        # Assuming all columns are features to scale
        data_scaled = scaler.transform(data)
        X_new = pd.DataFrame(data_scaled, columns=data.columns)
        print(f"Data shape after scaling: {X_new.shape}")
    else:
        X_new = data
        print("No scaler applied to input data.")

    return X_new

def make_predictions(model, X_new):
    """
    Use the loaded model to make predictions on new data.

    Parameters:
    - model: Loaded machine learning model.
    - X_new (DataFrame): Preprocessed features.

    Returns:
    - predictions (ndarray): Array of prediction results.
    """
    print("Making predictions...")
    predictions = model.predict(X_new)
    print("Predictions completed.")
    return predictions

def save_predictions(predictions, output_path):
    """
    Save prediction results to a CSV file.

    Parameters:
    - predictions (ndarray): Array of prediction results.
    - output_path (str): Path to save the predictions CSV.
    """
    print(f"Saving predictions to: {output_path}")
    df_predictions = pd.DataFrame(predictions, columns=['Predicted_RUL'])
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def main(args):
    """
    Main function to execute the prediction workflow.

    Parameters:
    - args: Parsed command-line arguments.
    """
    try:
        # Load the model
        if args.model_source == 'joblib':
            if not args.model_path:
                raise ValueError("model_path must be provided when model_source is 'joblib'.")
            model = load_model_joblib(args.model_path)
        elif args.model_source == 'mlflow':
            if not args.model_name:
                raise ValueError("model_name must be provided when model_source is 'mlflow'.")
            model = load_model_mlflow(args.model_name, args.run_id)
        else:
            raise ValueError("model_source must be either 'joblib' or 'mlflow'.")

        # Load and preprocess new data
        X_new = load_and_preprocess_data(args.input_data, args.scaler_path)

        # Make predictions
        predictions = make_predictions(model, X_new)

        # Save predictions
        save_predictions(predictions, args.output_path)
    except FileNotFoundError as fnf_error:
        print(f"File Not Found Error: {fnf_error}")
    except ValueError as val_error:
        print(f"Value Error: {val_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make predictions using the trained ML model.")

    # Model loading options
    parser.add_argument('--model_source', type=str, choices=['joblib', 'mlflow'], required=True,
                        help="Source of the model: 'joblib' or 'mlflow'.")
    parser.add_argument('--model_path', type=str,
                        help="Path to the joblib model file (required if model_source is 'joblib').")
    parser.add_argument('--model_name', type=str,
                        help="Name of the MLflow registered model (required if model_source is 'mlflow').")
    parser.add_argument('--run_id', type=str, default=None,
                        help="Specific MLflow run ID to load the model from (optional).")

    # Data input options
    parser.add_argument('--input_data', type=str, required=True,
                        help="Path to the new input data CSV file.")
    parser.add_argument('--scaler_path', type=str, default=None,
                        help="Path to the scaler file if scaling is required.")

    # Output options
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path to save the predictions CSV file.")

    args = parser.parse_args()

    main(args)
