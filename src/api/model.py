import joblib
import os
import pandas as pd
from typing import Optional

class PredictiveMaintenanceModel:
    """
    A class to handle loading the machine learning model and scaler,
    preprocessing input data, and making predictions for predictive maintenance.
    """
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Initializes the PredictiveMaintenanceModel by loading the model and scaler.
        
        Parameters:
        - model_path (str): Path to the trained machine learning model (.pkl file).
        - scaler_path (str, optional): Path to the scaler (.pkl file) used during training.
                                         Required if scaling was applied to the training data.
        
        Raises:
        - FileNotFoundError: If the model or scaler file does not exist at the specified path.
        - Exception: If there is an error loading the model or scaler.
        """
        # Load the trained model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}.")
        except Exception as e:
            raise Exception(f"An error occurred while loading the model: {e}")
        
        # Load the scaler if a path is provided
        if scaler_path:
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"Scaler loaded successfully from {scaler_path}.")
            except Exception as e:
                raise Exception(f"An error occurred while loading the scaler: {e}")
        else:
            self.scaler = None
            print("No scaler provided. Skipping scaling step.")

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the input data by applying scaling if a scaler is loaded.
        
        Parameters:
        - data (pd.DataFrame): Raw input features for prediction.
        
        Returns:
        - pd.DataFrame: Preprocessed (scaled) input features ready for prediction.
        
        Raises:
        - Exception: If scaling fails.
        """
        if self.scaler:
            try:
                # Apply the scaler to the input data
                scaled_data = self.scaler.transform(data)
                preprocessed_data = pd.DataFrame(scaled_data, columns=data.columns)
                print("Data scaling successful.")
                return preprocessed_data
            except Exception as e:
                raise Exception(f"An error occurred during data scaling: {e}")
        else:
            # If no scaler is provided, return the data as-is
            print("No scaling applied to data.")
            return data

    def predict(self, data: pd.DataFrame) -> float:
        """
        Makes a prediction using the loaded machine learning model.
        
        Parameters:
        - data (pd.DataFrame): Preprocessed input features.
        
        Returns:
        - float: Predicted Remaining Useful Life (RUL).
        
        Raises:
        - Exception: If prediction fails.
        """
        try:
            # Preprocess the data
            preprocessed_data = self.preprocess(data)
            
            # Make prediction
            prediction = self.model.predict(preprocessed_data)[0]
            print(f"Prediction successful: {prediction}")
            return prediction
        except Exception as e:
            raise Exception(f"An error occurred during prediction: {e}")
