import os
import argparse
import logging
from src.api.utils import clean_input_data

# --------------------
# Logging Configuration
# --------------------

# Configure logging to output logs to both a file and the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clean_new_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main(args):
    """
    Main function to execute the data cleaning workflow.

    Parameters:
    - args: Parsed command-line arguments.
    """
    try:
        logger.info("Starting data cleaning process.")
        
        # Clean input data using the utility function
        clean_input_data(
            input_path=args.input_csv,
            output_path=args.output_csv,
            drop_columns=args.drop_columns,
            handle_missing=args.handle_missing,
            impute_strategy=args.impute_strategy
        )
        
        logger.info("Data cleaning completed successfully.")
    
    except FileNotFoundError as fnf_error:
        logger.error(f"File Not Found Error: {fnf_error}")
    except ValueError as val_error:
        logger.error(f"Value Error: {val_error}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean new input data for predictive maintenance.")
    
    # Input and Output file paths
    parser.add_argument('--input_csv', type=str, required=True,
                        help="Path to the raw input data CSV file.")
    parser.add_argument('--output_csv', type=str, required=True,
                        help="Path to save the cleaned data CSV file.")
    
    # Data cleaning options
    parser.add_argument('--drop_columns', type=str, nargs='*', default=None,
                        help="List of column names to drop from the data.")
    parser.add_argument('--handle_missing', type=str, choices=['drop', 'impute'], default='drop',
                        help="Strategy to handle missing values: 'drop' or 'impute'.")
    parser.add_argument('--impute_strategy', type=str, choices=['mean', 'median', 'most_frequent'], default='mean',
                        help="Imputation strategy to use if handle_missing is 'impute'.")
    
    args = parser.parse_args()
    
    main(args)
