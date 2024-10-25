# Perform_AI.src.data_loader.files_extracting.py

import os
import logging
import pandas as pd
from .get_full_path import get_full_path
from .files_saving import FileSaver
from params import CLOUD_ON, BUCKET_NAME, USER_DATA_FILE


class FileLoader:

    """
    A class responsible for loading csv files of workout data, activities, TSS metrics, and nutrition information.
    """
    def __init__(self):
        """Initialize the FileLoader with the default file path and logging configuration."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.file_path = 'data/processed/csv'


    def _load_csv(self, file_path, name, index=None):
        """
        Loads the given dataframe to a CSV file.

        Args:
            file_path (str): The directory path to save the CSV.
            name (str): The name of the CSV file (without extension).
            index (int or None): Column to set as index in the dataframe (default is None).

        Returns:
            pd.DataFrame or bool: Loaded dataframe if successful, otherwise False.
        """
        try:
            # Set the full file path based on whether cloud storage is used
            full_path = f"s3://{BUCKET_NAME}/{file_path}/{name}.csv" if CLOUD_ON == 'yes' else os.path.join(get_full_path(file_path), f"{name}.csv")

            df = pd.read_csv(full_path, index_col=index)
            logging.info(f"{name.replace('_', ' ').title()} dataframe loaded successfully at {full_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading dataframe {name}: {e}")
            return False

    def load_initial_uploaded_workout_csv(self, name, file_path='data/raw/csv'):
        """
        Loads the workouts dataframe that the user uploads online.

        Args:
            name (str): The name of the CSV File (without extension).
            file_path (str): a special path different (for local only) from the default one.
            index (bool): Whether to include the dataframe index in the CSV file (default is False).
        """
        return self._load_csv(file_path, name)

    def load_csv_files(self, w_df=None, a_df=None, df=None, foods_df=None, file_path=None): # TODO
        """
        Loads final CSV files for workouts, activities, foods, and final data.

        Args:
            w_df (pd.DataFrame, optional): Workouts dataframe.
            a_df (pd.DataFrame, optional): Activities dataframe.
            df (pd.DataFrame, optional): Final merged dataframe.
            foods_df (pd.DataFrame, optional): Foods dataframe.
            file_path (str, optional): Custom file path for loading the CSVs.
        """
        for name, data in zip(['final_df', 'foods_df', 'workouts_df', 'activities_df'], [df, foods_df, w_df, a_df]):
            if data is not None:
                index = name in ['final_df', 'foods_df']  # Include index only for 'final_df' and 'foods_df'
                self._load_csv(self.file_path if file_path is None else file_path, data, name, index)

    def load_tss_values_for_dashboard(self, file_path=None):
        """
        Loads TSS, ATL, CTL, and TSB metrics for the dashboard.

        Args:
            file_path (str, optional): Custom file path for loading the CSVs (default is None).

        Returns:
            tuple: Dataframes for TSS, ATL, CTL, and TSB if successful, otherwise False.

        """
        names = ['tss', 'atl', 'ctl', 'tsb']
        dataframes = {
            name: self._load_csv(self.file_path if file_path is None else file_path, name, index=0)
            for name in names
        }
        return dataframes['tss'], dataframes['atl'], dataframes['ctl'], dataframes['tsb']

    def load_during_process(self, work_df, acti_df, file_path=None): # TODO
        """
        Save intermediate workout and activity dataframes during the processing step.

        Args:
            work_df (pd.DataFrame): Workouts dataframe to be saved.
            acti_df (pd.DataFrame): Activities dataframe to be saved.
            file_path (str, optional): Custom file path for saving the CSVs.
        """
        for name, data in zip(['workouts_to_process_df', 'activities_to_process_df'], [work_df, acti_df]):
            if data is not None:
                self._load_csv(self.file_path if file_path is None else file_path, data, name)

    def load_user_nutrition(self, file_path=None):
        """
        Save the user's nutrition data.

        Args:
            nutrition_df (pd.DataFrame): The nutrition dataframe to be saved.
            file_path (str, optional): Custom file path for saving the CSV.
        """
        df = self._load_csv(file_path or self.file_path, 'user_nutrition')
        if df is False:
            columns =   ['Timestamp', 'Meal', 'Food', 'Units', 'Grams per Unit', 'Total Grams', 'Calories', 'Fat', 'Saturated Fats',
                        'Monounsaturated Fats', 'Polyunsaturated Fats', 'Carbohydrates', 'Sugars', 'Protein', 'Dietary Fiber']
            df = pd.DataFrame(columns=columns)
            logging.info("Initialized a new nutrition DataFrame with predefined columns.")
            FileSaver().save_user_nutrition(nutrition_df=df)
        return df

    def load_user_data(self, file_path=USER_DATA_FILE):
        """
        Loads the user data in a special hidden folder.

        Args:
            user_data (pd.DataFrame): The user information to be saved: BMR, passive calories, etc.
            file_path (str): a special path hidden and different (for local only) from the default one.
            index (bool): Whether to include the dataframe index in the CSV file (default is False).
        """
        return self._load_csv(self.file_path if file_path is None else file_path, 'user_data')
