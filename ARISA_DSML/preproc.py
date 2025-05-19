import pandas as pd
import os
import numpy as np
from loguru import logger
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from kaggle.api.kaggle_api_extended import KaggleApi
from ARISA_DSML.config import DATASET, PROCESSED_DATA_DIR, RAW_DATA_DIR, target



sys.path.append(os.path.abspath(".."))


def get_raw_data(dataset: str = DATASET) -> None:
    api = KaggleApi()
    api.authenticate()

    download_folder = Path(RAW_DATA_DIR)
    download_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading data to: {download_folder}")

    # Download specific files from the dataset
    api.dataset_download_files(
        dataset=dataset,
        path=str(download_folder),
        unzip=True,
        quiet=False,
    )

    logger.info("Dataset download and extraction completed successfully")


# Creating function to change the text to numerical values
def preprocess_categorical_data(df):

    # Apply mappings for categorical variables
    mappings = {
        'Physical_Activity': {'Low': 3, 'Moderate': 2, 'High': 1},
        'Dietary_Habits': {'Unhealthy': 3, 'Moderate': 2, 'Healthy': 1},
        'Family_History': {'No': 0, 'Yes': 1}
    }

# Define the columns for one-hot encoding
    one_hot_columns = ['Gender']

    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)

    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=one_hot_columns)

    return df


def scale_features(X):
    # Remove zero-variance columns
    std_devs = X.std()
    zero_variance_cols = std_devs[std_devs == 0].index.tolist()
    if zero_variance_cols:
        X = X.drop(columns=zero_variance_cols)
        print(f"Removed zero-variance columns: {zero_variance_cols}")

    # Handle NaN/Inf values (replace with 0 or drop rows/columns)
    X = X.fillna(0)  # Replace NaNs with 0; adjust as needed
    X = X.replace([np.inf, -np.inf], 0)

    # Scale features
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)

    # Convert to DataFrame
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)

    return X_scaled, scaler


def preprocess_df(file: str | Path) -> str | Path:
    """Preprocess datasets by handling categorical data, scaling features, and saving results."""
    # Read raw data
    df = pd.read_excel(file, sheet_name='Sheet1')

    # Separate features from target
    X = df.drop(columns=[target, 'id'])  # Remove target and id columns from features
    y = df[target]

    # Process categorical data
    X_processed = preprocess_categorical_data(X)

    # Scale numerical features
    X_scaled, _ = scale_features(X_processed)  # We ignore the scaler return value for now
    X_scaled[target] = y.values

    # Ensure output directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save processed data
    _, file_name = os.path.split(file)
    outfile_path = PROCESSED_DATA_DIR / file_name
    X_scaled.to_csv(outfile_path, index=False)

    return outfile_path


if __name__ == "__main__":
    # get the train and test sets from default location
    logger.info("getting datasets")
    get_raw_data()

    # preprocess both sets
    logger.info("preprocessing train_cholesterol.xlsx")
    preprocess_df(RAW_DATA_DIR / "train_cholesterol.xlsx")
    logger.info("preprocessing test_cholesterol.xlsx")
    preprocess_df(RAW_DATA_DIR / "test_cholesterol.xlsx")
