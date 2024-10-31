# Perform_AI.src.data_loader.files_saving.py

import os
import logging
from .get_full_path import get_full_path
from params import CLOUD_ON, BUCKET_NAME, USER_DATA_FILE
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

    # def save_initial_uploaded_workout_csv(self, workouts, name, file_path='data/raw/csv'):
    #     """
    #     Saves the workouts dataframe that the user uploads online.

    #     Args:
    #         workouts (pd.Dataframe): The dataframe to be saved.
    #         name (str): The name of the CSV File (without extension). -> 'upload_new_data_workouts_' + st.session_state['username']
    #         file_path (str): a special path different (for local only) from the default one.
    #         index (bool): Whether to include the dataframe index in the CSV file (default is False).
    #     """
    #     self._save_csv(file_path, workouts, name)

    def save_raw_and_final_dataframes(self, w_df=None, a_df=None, df=None, foods_df=None, file_path=None):
        """
        Save raw and final dataframes for workouts, activities, foods, and final data.

        Args:
            w_df (pd.DataFrame, optional): Workouts dataframe.
            a_df (pd.DataFrame, optional): Activities dataframe.
            df (pd.DataFrame, optional): Final merged dataframe.
            foods_df (pd.DataFrame, optional): Foods dataframe.
            file_path (str, optional): Custom file path for saving the CSVs.
        """
        for name, data in zip(['final_df', 'foods_df', 'workouts_df', 'activities_df'], [df, foods_df, w_df, a_df]):
            if data is not None:
                index = name in ['final_df', 'foods_df']  # Include index only for 'final_df' and 'foods_df'
                self._save_csv(self.file_path if file_path is None else file_path, data, name, index)

    # def save_tss_values_for_dashboard(self, tss, atl, ctl, tsb, file_path=None):
    #     """
    #     Save TSS, ATL, CTL, and TSB metrics for the dashboard.

    #     Args:
    #         tss (pd.DataFrame): Training Stress Score (TSS) dataframe.
    #         atl (pd.DataFrame): Acute Training Load (ATL) dataframe.
    #         ctl (pd.DataFrame): Chronic Training Load (CTL) dataframe.
    #         tsb (pd.DataFrame): Training Stress Balance (TSB) dataframe.
    #         file_path (str, optional): Custom file path for saving the CSVs.
    #     """
    #     for name, data in zip(['tss', 'atl', 'ctl', 'tsb'], [tss, atl, ctl, tsb]):
    #         if data is not None:
    #             self._save_csv(self.file_path if file_path is None else file_path, data, name, index=True)




    # def save_during_process(self, file_path=None, **kwargs):
    #     """
    #     Save tmp workout and activity dataframes during the processing step.

    #     Args:
    #         **kwargs (pd.DataFrame): Workouts and/or Activities dataframe to be saved.
    #         file_path (str, optional): Custom file path for saving the CSVs.
    #     """
    #     for name, data in zip(['workouts_tmp_df', 'activities_tmp_df'], [kwargs.get('workouts_tmp_df'), kwargs.get('activities_tmp_df')]):
    #         if data is not None:
    #             self._save_csv(file_path or self.file_path, data, name, index=True)

    # def save_user_nutrition(self, nutrition_df, file_path=None):
    #     """
    #     Save the user's nutrition data.

    #     Args:
    #         nutrition_df (pd.DataFrame): The nutrition dataframe to be saved.
    #         file_path (str, optional): Custom file path for saving the CSV.
    #     """
    #     self._save_csv(self.file_path if file_path is None else file_path, nutrition_df, 'user_nutrition')

    # def save_user_data(self, user_data, file_path=USER_DATA_FILE):
    #     """
    #     Saves the user data in a special hidden folder.

    #     Args:
    #         user_data (pd.DataFrame): The user information to be saved: BMR, passive calories, etc.
    #         file_path (str): a special path hidden and different (for local only) from the default one.
    #         index (bool): Not passed, default is False
    #     """
    #     self._save_csv(self.file_path if file_path is None else file_path, user_data, 'user_data')
