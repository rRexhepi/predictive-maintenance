import os
import logging
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
import pandas as pd

from src.api.schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse
from src.api.model import PredictiveMaintenanceModel
from src.api.utils import clean_input_data, save_predictions, validate_data

# --------------------
# Logging Configuration
# --------------------

# Configure logging to output logs to both a file and the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------
# FastAPI Initialization
# --------------------

app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting Remaining Useful Life (RUL) using a trained machine learning model.",
    version="1.0.0"
)

# --------------------
# Model and Scaler Loading
# --------------------

# Retrieve model and scaler paths from environment variables or use default paths
MODEL_PATH = os.getenv("MODEL_PATH", "models/rf_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

# Initialize the PredictiveMaintenanceModel
try:
    model = PredictiveMaintenanceModel(model_path=MODEL_PATH, scaler_path=SCALER_PATH)
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or scaler: {e}")
    model = None  # Set to None to handle errors in endpoints

# --------------------
# API Key Authentication Setup
# --------------------

# Retrieve API key from environment variables for security
API_KEY = os.getenv("API_KEY", "your-default-api-key")  # Replace with a secure key in production
API_KEY_NAME = "access_token"  # Header name to expect the API key

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key: Optional[str] = Security(api_key_header)):
    """
    Validates the API key provided in the request header.

    Parameters:
    - api_key (str): API key from the request header.

    Returns:
    - str: Validated API key.

    Raises:
    - HTTPException: If the API key is invalid or missing.
    """
    if api_key == API_KEY:
        return api_key
    else:
        logger.warning(f"Unauthorized access attempt with API key: {api_key}")
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# --------------------
# Prediction Endpoints
# --------------------

@app.post("/predict", response_model=PredictResponse, dependencies=[Security(get_api_key)])
def predict(request: PredictRequest):
    """
    Endpoint to make a single prediction for Remaining Useful Life (RUL).

    Parameters:
    - request (PredictRequest): Input features for prediction.

    Returns:
    - PredictResponse: The predicted RUL.

    Raises:
    - HTTPException: If the model is not loaded or prediction fails.
    """
    if model is None:
        logger.error("Prediction attempted without a loaded model.")
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        # Convert the incoming request to a pandas DataFrame
        input_data = pd.DataFrame([request.dict()])
        logger.info(f"Received prediction request: {input_data.to_dict(orient='records')}")

        # Save input data to a temporary file
        temp_input_path = "temp_input.csv"
        input_data.to_csv(temp_input_path, index=False)
        logger.info(f"Input data saved to temporary file: {temp_input_path}")

        # Clean input data using utils
        temp_clean_path = "temp_clean_data.csv"
        clean_input_data(
            input_path=temp_input_path,
            output_path=temp_clean_path,
            drop_columns=['RUL'],
            handle_missing='impute',
            impute_strategy='median'
        )
        logger.info(f"Input data cleaned and saved to: {temp_clean_path}")

        # Load cleaned data
        cleaned_data = pd.read_csv(temp_clean_path)
        logger.info(f"Cleaned data loaded. Shape: {cleaned_data.shape}")

        # Validate data
        required_features = ['feature1', 'feature2', 'feature3', 'temperature', 'pressure']
        validate_data(cleaned_data, required_features)

        # Make prediction
        prediction = model.predict(cleaned_data)
        logger.info(f"Prediction made: {prediction}")

        # Save predictions
        predictions_output_path = "predictions/predictions.csv"
        save_predictions(predictions=[prediction], output_path=predictions_output_path)
        logger.info(f"Prediction saved to: {predictions_output_path}")

        # Clean up temporary files
        os.remove(temp_input_path)
        os.remove(temp_clean_path)
        logger.info("Temporary files removed.")

        return PredictResponse(predicted_rul=prediction)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictResponse, dependencies=[Security(get_api_key)])
def batch_predict(request: BatchPredictRequest):
    """
    Endpoint to make batch predictions for Remaining Useful Life (RUL).

    Parameters:
    - request (BatchPredictRequest): A list of input features for prediction.

    Returns:
    - BatchPredictResponse: A list of predicted RULs.

    Raises:
    - HTTPException: If the model is not loaded or prediction fails.
    """
    if model is None:
        logger.error("Batch prediction attempted without a loaded model.")
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        # Convert the list of incoming requests to a pandas DataFrame
        input_data = pd.DataFrame([item.dict() for item in request.data])
        logger.info(f"Received batch prediction request with {len(input_data)} records.")

        # Save input data to a temporary file
        temp_input_path = "temp_batch_input.csv"
        input_data.to_csv(temp_input_path, index=False)
        logger.info(f"Batch input data saved to temporary file: {temp_input_path}")

        # Clean input data using utils
        temp_clean_path = "temp_batch_clean_data.csv"
        clean_input_data(
            input_path=temp_input_path,
            output_path=temp_clean_path,
            drop_columns=['RUL'],
            handle_missing='impute',
            impute_strategy='median'
        )
        logger.info(f"Batch input data cleaned and saved to: {temp_clean_path}")

        # Load cleaned data
        cleaned_data = pd.read_csv(temp_clean_path)
        logger.info(f"Cleaned batch data loaded. Shape: {cleaned_data.shape}")

        # Validate data
        required_features = ['feature1', 'feature2', 'feature3', 'temperature', 'pressure']
        validate_data(cleaned_data, required_features)

        # Make predictions
        predictions = model.predict(cleaned_data)
        logger.info(f"Batch predictions made: {predictions}")

        # Save predictions
        predictions_output_path = "predictions/batch_predictions.csv"
        save_predictions(predictions=predictions, output_path=predictions_output_path)
        logger.info(f"Batch predictions saved to: {predictions_output_path}")

        # Clean up temporary files
        os.remove(temp_input_path)
        os.remove(temp_clean_path)
        logger.info("Temporary batch files removed.")

        return BatchPredictResponse(predictions=predictions.tolist())

    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# --------------------
# Root Endpoint
# --------------------

@app.get("/", include_in_schema=False)
def root():
    """
    Root endpoint providing a simple welcome message.
    """
    return {"message": "Welcome to the Predictive Maintenance API. Visit /docs for API documentation."}

# --------------------
# Startup and Shutdown Events
# --------------------

@app.on_event("startup")
def startup_event():
    """
    Actions to perform on application startup.
    """
    if model is None:
        logger.critical("Application startup failed: Model not loaded.")
        raise RuntimeError("Model not loaded.")
    logger.info("Predictive Maintenance API is up and running.")

@app.on_event("shutdown")
def shutdown_event():
    """
    Actions to perform on application shutdown.
    """
    logger.info("Predictive Maintenance API is shutting down.")
    
    # Removal of any leftover temporary files
    temp_files = [
        "temp_input.csv",
        "temp_clean_data.csv",
        "temp_batch_input.csv",
        "temp_batch_clean_data.csv"
    ]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logger.info(f"Removed leftover temporary file: {temp_file}")
