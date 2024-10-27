# Perform_AI.src.data_loader.files_extracting.py

import os
import logging
import pandas as pd
from .get_full_path import get_full_path
from .files_saving import FileSaver
from params import CLOUD_ON, BUCKET_NAME, USER_DATA_FILE


class FileLoader:
    """
    A class responsible for loading and saving CSV files related to workout, activities, TSS metrics, and nutrition data.
    """
    def __init__(self):
        """Initialize the FileLoader with the default file path and logging configuration."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.file_path = 'data/processed/csv'
        self.activities_raw = None
        self.workouts_raw = None

        self.activities_processed = None
        self.workouts_processed = None
        self.final = None

        self.foods = None

    def _load_csv(self, file_path, name, index=None, **kwargs):
        """
        Loads a CSV file into a DataFrame.

        Args:
            file_path (str): The directory path of the CSV file.
            name (str): The name of the CSV file (without extension).
            index (int or None): Column to set as index in the DataFrame (default is None).

        Returns:
            pd.DataFrame or bool: Loaded DataFrame if successful, otherwise False.
        """
        try:
            full_path = f"s3://{BUCKET_NAME}/{file_path}/{name}.csv" if CLOUD_ON == 'yes' else os.path.join(get_full_path(file_path), f"{name}.csv")
            df = pd.read_csv(full_path, index_col=index, **kwargs)
            logging.info(f"{name.replace('_', ' ').title()} dataframe loaded successfully from {full_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading dataframe {name}: {e}")
            return False

    def load_initial_uploaded_workout_csv(self, name, file_path='data/raw/csv'):
        """
        Loads the workouts DataFrame that the user uploads online.

        Args:
            name (str): The name of the CSV file (without extension).
            file_path (str): Path to the directory containing the CSV file (default is 'data/raw/csv').

        Returns:
            pd.DataFrame or bool: Loaded DataFrame if successful, otherwise False.
        """
        return self._load_csv(file_path, name)

    def load_raw_and_final_dataframes(self, file_path=None):
        """
        Loads final Dataframes for workouts, activities and final merged data.
        Loads raw Foods Dataframe.

        Args:
            file_path (str, optional): Custom file path ( data/raw/csv ) for loading the foods_df.

        Returns:
            tuple: Loaded DataFrames for final_df, workouts_df, and activities_df if no file_path is provided;
                   otherwise, only foods_df if file_path is specified.
        """
        names = ['final_df', 'workouts_df', 'activities_df', 'foods_df']
        dataframes = {name: self._load_csv(file_path or self.file_path, name) for name in names}

        if file_path:
            self.activities_raw = dataframes['activities_df']
            self.workouts_raw =  dataframes['workouts_df']
            self.foods = dataframes['foods_df']
        else:
            self.activities_processed = dataframes['activities_df']
            self.workouts_processed = dataframes['workouts_df']
            self.final = dataframes['final_df']

    def load_final_with_no_na_filter(self):
        """
        Load the 'final_df.csv' with na_filter set to False.

        Returns:
            pd.DataFrame: Loaded DataFrame with na_filter=False.
        """
        return self._load_csv(self.file_path, 'final_df', index=0, na_filter=False)

    def load_tss_values_for_dashboard(self, file_path=None):
        """
        Loads TSS, ATL, CTL, and TSB metrics for dashboard display.

        Args:
            file_path (str, optional): Custom file path for loading the CSV files (default is None).

        Returns:
            tuple: Loaded DataFrames for TSS, ATL, CTL, and TSB if successful, otherwise False.
        """
        names = ['tss', 'atl', 'ctl', 'tsb']
        dataframes = {name: self._load_csv(file_path or self.file_path, name, index=0) for name in names} # NOTE: self.file_path if file_path is None else file_path
        return dataframes['tss'], dataframes['atl'], dataframes['ctl'], dataframes['tsb']

    def load_during_process(self, file_path=None):
        """
        Loads intermediate workout and activity data during the processing stage.

        Args:
            file_path (str, optional): Custom file path for loading the CSV files (default is None).

        Returns:
            tuple: Loaded DataFrames for workouts_to_process_df and activities_to_process_df.
        """
        names = ['workouts_to_process_df', 'activities_to_process_df']
        dataframes = {name: self._load_csv(file_path or self.file_path, name) for name in names}
        return dataframes['workouts_to_process_df'], dataframes['activities_to_process_df']

    def load_user_nutrition(self, file_path=None):
        """
        Loads or initializes the user's nutrition data.

        Args:
            file_path (str, optional): Custom file path for loading or saving the CSV file (default is None).

        Returns:
            pd.DataFrame: Loaded or initialized nutrition DataFrame.
        """
        df = self._load_csv(file_path or self.file_path, 'user_nutrition')
        if df is False:
            columns = ['Timestamp', 'Meal', 'Food', 'Units', 'Grams per Unit', 'Total Grams', 'Calories', 'Fat',
                       'Saturated Fats', 'Monounsaturated Fats', 'Polyunsaturated Fats', 'Carbohydrates', 'Sugars',
                       'Protein', 'Dietary Fiber']
            df = pd.DataFrame(columns=columns)
            logging.info("Initialized a new nutrition DataFrame with predefined columns.")
            FileSaver().save_user_nutrition(nutrition_df=df)
        return df

    def load_user_data(self, file_path=USER_DATA_FILE):
        """
        Loads the user's data, including BMR and passive calories.

        Args:
            file_path (str): Custom path to the user's data file.

        Returns:
            pd.DataFrame or bool: Loaded DataFrame if successful, otherwise False.
        """
        return self._load_csv(file_path, 'user_data')

    def load_initial_csv_files(self):
        """
        Loads multiple workout and activity data files, merges them as necessary, and saves the combined files.

        Args:
            NOTE: NOT SURE -> key (str): Whether 'workouts' 'activities' 'foods'

        Returns:
            tuple: DataFrames for merged workouts and all-year activities.
        """

        dataframes_names = {
            'workouts': ['tp_workouts_2022-03-03_to_2023-03-03', 'tp_workouts_2023-03-03_to_2024-03-03', 'tp_workouts_2024-03-03_to_2025-03-03'],
            'activities': 'activities',
            'foods': [f"FOOD-DATA-GROUP{i}" for i in range(1,6)]
        }

        workouts_df = pd.concat([self._load_csv('data/raw/csv', name) for name in dataframes_names['workouts']], ignore_index=True)
        activities_df = self._load_csv('data/raw/csv', dataframes_names['activities'])
        foods = pd.concat([self._load_csv('data/raw/csv', name, index=0) for name in dataframes_names['foods']], ignore_index=True)

        # Define unwanted columns (adjust based on your needs)
        unwanted_columns = ['Unnamed: 0']  # You can add more if needed

        # Remove unwanted columns from all dataframes
        foods_df = foods.drop(columns=unwanted_columns, errors='ignore')

        FileSaver().save_csv_files(w_df = workouts_df, a_df=activities_df, foods_df=foods_df, file_path = 'data/raw/csv')
