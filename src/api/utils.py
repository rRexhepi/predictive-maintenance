# src/api/utils.py

import pandas as pd
import os
from typing import List, Optional
from sklearn.impute import SimpleImputer


def clean_input_data(
    input_path: str,
    output_path: str,
    drop_columns: Optional[List[str]] = None,
    handle_missing: str = "drop",
    impute_strategy: str = "mean"
) -> None:
    """
    Cleans the input CSV data by removing specified columns and handling missing values.

    Parameters:
    - input_path (str): Path to the original input CSV file.
    - output_path (str): Path to save the cleaned CSV file.
    - drop_columns (List[str], optional): List of column names to drop from the data.
    - handle_missing (str): Strategy to handle missing values. Options are 'drop' or 'impute'.
    - impute_strategy (str): Strategy to use for imputing missing values if handle_missing is 'impute'.
                             Options include 'mean', 'median', 'most_frequent', etc.

    Raises:
    - FileNotFoundError: If the input file does not exist.
    - ValueError: If an invalid handle_missing strategy is provided.
    - Exception: For any other errors during processing.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data file not found at {input_path}")

    try:
        # Load data
        df = pd.read_csv(input_path)
        print(f"Loaded data from {input_path}. Shape: {df.shape}")

        # Drop specified columns if any
        if drop_columns:
            missing_cols = [col for col in drop_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: Columns {missing_cols} not found in the data. They will be skipped.")
            df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)
            print(f"Dropped columns: {drop_columns}")
            print(f"Data shape after dropping columns: {df.shape}")

        # Handle missing values
        if handle_missing == "drop":
            initial_shape = df.shape
            df.dropna(inplace=True)
            dropped_rows = initial_shape[0] - df.shape[0]
            print(f"Dropped {dropped_rows} rows due to missing values.")
        elif handle_missing == "impute":
            imputer = SimpleImputer(strategy=impute_strategy)
            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            df = df_imputed
            print(f"Imputed missing values using '{impute_strategy}' strategy.")
        else:
            raise ValueError("handle_missing must be either 'drop' or 'impute'.")

        # Save cleaned data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}. Shape: {df.shape}")

    except ValueError as ve:
        print(f"ValueError during data cleaning: {ve}")
        raise ve
    except Exception as e:
        print(f"An error occurred during data cleaning: {e}")
        raise e


def save_predictions(
    predictions: List[float],
    output_path: str,
    input_identifier: Optional[List[str]] = None
) -> None:
    """
    Saves the prediction results to a CSV file.

    Parameters:
    - predictions (List[float]): List of predicted Remaining Useful Life (RUL) values.
    - output_path (str): Path to save the predictions CSV file.
    - input_identifier (List[str], optional): List of identifiers corresponding to each prediction.
                                             This could be IDs, timestamps, or any relevant identifiers.

    Raises:
    - ValueError: If input_identifier is provided but its length does not match predictions.
    - Exception: For any other errors during saving.
    """
    try:
        # Create DataFrame for predictions
        if input_identifier:
            if len(input_identifier) != len(predictions):
                raise ValueError("Length of input_identifier must match length of predictions.")
            df_predictions = pd.DataFrame({
                "Identifier": input_identifier,
                "Predicted_RUL": predictions
            })
        else:
            df_predictions = pd.DataFrame({
                "Predicted_RUL": predictions
            })

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        df_predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}. Number of predictions: {len(predictions)}")

    except ValueError as ve:
        print(f"ValueError during saving predictions: {ve}")
        raise ve
    except Exception as e:
        print(f"An error occurred while saving predictions: {e}")
        raise e
