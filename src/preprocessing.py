import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_data(file_name):
    """
    Load the dataset from the specified file name.

    Parameters:
    - file_name (str): Name of the data file.

    Returns:
    - df (DataFrame): Loaded DataFrame.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the data directory
    data_dir = os.path.join(script_dir, '..', 'data')
    # Construct the full file path
    file_path = os.path.join(data_dir, file_name)

    # Verify that the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")

    # Define column names as per dataset documentation
    column_names = ['Unit', 'Time', 'Setting1', 'Setting2', 'Setting3'] + [f'Sensor{i}' for i in range(1, 22)]
    # Use raw string for the separator
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=column_names)
    return df

def handle_outliers(df, sensor_columns):
    """
    Cap outliers at the 1st and 99th percentiles for sensor readings.

    Parameters:
    - df (DataFrame): Input DataFrame.
    - sensor_columns (list): List of sensor column names.

    Returns:
    - df (DataFrame): DataFrame with outliers handled.
    """
    for col in sensor_columns:
        lower_limit = df[col].quantile(0.01)
        upper_limit = df[col].quantile(0.99)
        df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])
        df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
    return df

def add_rul(df):
    """
    Add the Remaining Useful Life (RUL) column to the DataFrame.

    Parameters:
    - df (DataFrame): Input DataFrame.

    Returns:
    - df (DataFrame): DataFrame with RUL column added.
    """
    max_cycles = df.groupby('Unit')['Time'].max().reset_index()
    max_cycles.columns = ['Unit', 'MaxTime']
    df = df.merge(max_cycles, on='Unit')
    df['RUL'] = df['MaxTime'] - df['Time']
    df.drop('MaxTime', axis=1, inplace=True)
    return df

def remove_irrelevant_features(df):
    """
    Remove constant and highly correlated sensor features.

    Parameters:
    - df (DataFrame): Input DataFrame.

    Returns:
    - df (DataFrame): DataFrame with irrelevant features removed.
    """
    # Constant sensors identified during EDA
    constant_sensors = ['Sensor1', 'Sensor5', 'Sensor6', 'Sensor10', 'Sensor16', 'Sensor18', 'Sensor19']
    # Highly correlated sensors identified during EDA
    correlated_sensors = ['Sensor15', 'Sensor17']  # Modify based on your EDA results
    drop_columns = constant_sensors + correlated_sensors
    df.drop(columns=drop_columns, inplace=True)
    return df

def create_lag_features(df, sensors, lags):
    """
    Create lag features for specified sensors.

    Parameters:
    - df (DataFrame): Input DataFrame.
    - sensors (list): List of sensor column names.
    - lags (list): List of lag periods.

    Returns:
    - df (DataFrame): DataFrame with lag features added.
    """
    for sensor in sensors:
        for lag in lags:
            df[f'{sensor}_lag{lag}'] = df.groupby('Unit')[sensor].shift(lag)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)  # Reset index after dropping rows
    return df

def scale_features(df, feature_columns):
    """
    Scale features using StandardScaler.

    Parameters:
    - df (DataFrame): Input DataFrame.
    - feature_columns (list): List of feature column names.

    Returns:
    - scaled_df (DataFrame): DataFrame with scaled features.
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[feature_columns])
    scaled_df = pd.DataFrame(scaled_array, columns=feature_columns, index=df.index)

    # Save the scaler
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)

    return scaled_df

def preprocess_data():
    """
    Execute the preprocessing pipeline:
    - Load data
    - Handle outliers
    - Add RUL
    - Remove irrelevant features
    - Create lag features
    - Scale features
    - Split data into training and validation sets
    - Save preprocessed data
    """
    # Load data
    df_train = load_data('train_FD001.txt')

    # Identify sensor columns
    sensor_columns = [col for col in df_train.columns if 'Sensor' in col]

    # Handle outliers
    df_train = handle_outliers(df_train, sensor_columns)

    # Add RUL
    df_train = add_rul(df_train)

    # Remove irrelevant features
    df_train = remove_irrelevant_features(df_train)

    # Update sensor columns after removal
    sensor_columns = [col for col in df_train.columns if 'Sensor' in col]

    # Feature Engineering: Create lag features
    df_train = create_lag_features(df_train, sensor_columns, lags=[1, 2, 3])

    # Define features and target
    feature_columns = [col for col in df_train.columns if col not in ['Unit', 'Time', 'RUL']]
    X = df_train[feature_columns]
    y = df_train['RUL']

    # Scale features
    X_scaled = scale_features(df_train, feature_columns)
    df_train[feature_columns] = X_scaled

    # Split data
    units = df_train['Unit'].unique()
    units_train, units_val = train_test_split(units, test_size=0.2, random_state=42)

    # Training set
    train_indices = df_train['Unit'].isin(units_train)
    X_train = df_train.loc[train_indices, feature_columns]
    y_train = df_train.loc[train_indices, 'RUL']

    # Validation set
    val_indices = df_train['Unit'].isin(units_val)
    X_val = df_train.loc[val_indices, feature_columns]
    y_val = df_train.loc[val_indices, 'RUL']

    # Ensure the 'data' directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Save preprocessed data
    train_data = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    val_data = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
    train_data.to_csv(os.path.join(data_dir, 'train_preprocessed.csv'), index=False)
    val_data.to_csv(os.path.join(data_dir, 'val_preprocessed.csv'), index=False)

if __name__ == '__main__':
    preprocess_data()
