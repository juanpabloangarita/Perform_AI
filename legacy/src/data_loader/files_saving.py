# Perform_AI.src.data_loader.files_saving.py

import os
import logging
from .get_full_path import get_full_path
from params import CLOUD_ON, BUCKET_NAME
import joblib
import tempfile
import boto3
import pandas as pd
s3 = boto3.client('s3')


class FileSaver:
    """
    A class responsible for saving csv files of workout data, activities, TSS metrics, and nutrition information.
    """
    def __init__(self):
        """Initialize the FileSaver with the default file path and logging configuration."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.file_path = 'data/processed/csv'

    def _save_csv(self, file_path, model_or_df, name, index=False):
        """
        Saves the given dataframe to a CSV file or trained models.

        Args:
            file_path (str): The directory path to save the CSV.
            df (pd.DataFrame): The dataframe to be saved.
            name (str): The name of the CSV file (without extension).
            index (bool): Whether to include the dataframe index in the CSV file (default is False).
        """
        extension, component = ('pkl', 'model') if file_path == 'data/processed/models' else ('csv', 'dataframe')
        try:
            # Set the full file path based on whether cloud storage is used
            full_path = f"s3://{BUCKET_NAME}/{file_path}/{name}.{extension}" if CLOUD_ON == 'yes' else os.path.join(get_full_path(file_path), f"{name}.{extension}")
            if component == 'model':
                if CLOUD_ON=='yes':
                    full_path = f"{file_path}/{name}.{extension}"
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                        joblib.dump(model_or_df, temp_file.name)  # Save temporarily
                        s3.upload_file(temp_file.name, BUCKET_NAME, full_path)
                else:
                    joblib.dump(model_or_df, full_path)
            else:
                model_or_df.to_csv(full_path, index=index, na_rep='')  # Save CSV with or without index
            # joblib.dump(model_or_df, full_path) if component == 'model' else model_or_df.to_csv(full_path, index=index, na_rep='')  # Save CSV with or without index
            logging.info(f"{name.replace('_', ' ').title()} {component} saved successfully at {full_path}")
        except Exception as e:
            logging.error(f"Error saving {component} {name}: {e}")

    def save_models(self, model, name, file_path='data/processed/models'):
        """Save the model using the _save_csv helper method."""
        self._save_csv(file_path, model, name)

    def save_dfs(self, dfs, dfs_names=None, file_path=None, name=None, index=False):
        """
        Save one or multiple dataframes to CSV files.

        This method can save either a single dataframe or a list of dataframes.
        If saving a list, the corresponding names must be provided in dfs_names.

        Args:
            dfs (Union[pd.DataFrame, list]): The dataframe(s) to be saved.
                Can be a single pandas DataFrame or a list of DataFrames.
            dfs_names (list, optional): The names of the CSV files to save each dataframe.
                Required if dfs is a list.
            file_path (str, optional): The directory path to save the CSV files.
                If not provided, defaults to the instance's file_path.
            name (str, optional): The name of the CSV file to save if a single dataframe is provided.
                Ignored if dfs is a list.
            index (bool): Whether to include the dataframe index in the CSV file (default is False).

        Raises:
            ValueError: If dfs is a list but dfs_names is not provided.
        """
        if isinstance(dfs, list):
            for list_name, data in zip(dfs_names, dfs):
                if data is not None:
                    self._save_csv(self.file_path if file_path is None else file_path, data, list_name, index=index)

        if isinstance(dfs, pd.DataFrame):
            self._save_csv(self.file_path if file_path is None else file_path, dfs, name, index=index)
