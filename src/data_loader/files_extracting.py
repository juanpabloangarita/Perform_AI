# Perform_AI.src.data_loader.files_extracting.py

import os
import logging
import pandas as pd
from .files_saving import FileSaver
from params import CLOUD_ON, BUCKET_NAME, USER_DATA_FILE
from .get_full_path import get_full_path
from src.tss_calculations import calculate_total_tss_and_metrics_from_tss
from src.data_loader.update_final_df_helper import (
    process_data_to_update,
    process_activity_dict,
    update_or_add_row,
    create_default_row
)
import joblib
import tempfile
import boto3
s3 = boto3.client('s3')


class FileLoader:
    """
    A class responsible for loading CSV files and Dataframes related to workout, activities, TSS metrics, and nutrition data.
    It exclusively saves the initial loaded csv files: workouts, activities and foods.
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
        self.workouts_tmp_df = None
        self.activities_tmp_df = None

    def _load_csv(self, file_path, name, index=None, **kwargs):
        """
        Loads : a CSV file into a DataFrame OR dataframes OR models

        Args:
            file_path (str): The directory path of the CSV file.
            name (str): The name of the CSV file (without extension).
            index (int or None): Column to set as index in the DataFrame (default is None).

        Returns:
            pd.DataFrame or bool: Loaded DataFrame if successful, otherwise False.
        """
        extension, component = ('pkl', 'model') if file_path == 'data/processed/models' else ('csv', 'dataframe')
        try:
            full_path = f"s3://{BUCKET_NAME}/{file_path}/{name}.{extension}" if CLOUD_ON == 'yes' else os.path.join(get_full_path(file_path), f"{name}.{extension}")
            if component == 'model':
                if CLOUD_ON=='yes':
                    full_path = f"{file_path}/{name}.{extension}"
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                        s3.download_file(BUCKET_NAME, full_path, temp_file.name)
                        model_or_df = joblib.load(temp_file.name)
                else:
                    model_or_df = joblib.load(full_path)
            else:
                model_or_df = pd.read_csv(full_path, index_col=index, **kwargs)
            # model_or_df = joblib.load(full_path) if component == 'model' else pd.read_csv(full_path, index_col=index, **kwargs)
            logging.info(f"{name.replace('_', ' ').title()} {component} loaded successfully from {full_path}")
            return model_or_df
        except Exception as e:
            logging.error(f"Error loading {component} {name}: {e}")
            return False

    def load_models(self, name, file_path='data/processed/models'):
            """
            Load a model from the specified file path using the _load_csv method.

            Args:
                name (str): The name of the model (without extension).
                file_path (str): The directory path of the model files.

            Returns:
                Loaded model or None: Returns the loaded model or None if not found.
            """
            model = self._load_csv(file_path, name)
            if model is False:
                logging.warning(f"Model {name} not found. Training a new one.")
                return None
            return model

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
        names = ['workouts_df', 'activities_df', 'foods_df'] if file_path == 'data/raw/csv' else ['final_df', 'workouts_df', 'activities_df', 'foods_df']
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

    def update_final_df(self, from_where, time_added=None, data_to_update=None):
            """
            Load the final DataFrame and update it based on the source of the data.

            Args:
                from_where (str): Source of the update.
                time_added (str, optional): Time to add the data.
                data_to_update (various): Data to be updated in the DataFrame.

            Returns:
                pd.DataFrame: Updated DataFrame.
            """
            df = self.load_final_with_no_na_filter()

            if from_where in ['home', 'plan_my_day']:
                return df

            elif from_where == 'training_peaks' and isinstance(data_to_update, pd.DataFrame):
                df = process_data_to_update(df, data_to_update)

            elif from_where == 'training_peaks' and isinstance(data_to_update, list):
                df = process_activity_dict(df, data_to_update)

            elif from_where == 'plan_my_week':
                updates = {
                    'TimeTotalInHours': pd.to_numeric(data_to_update.get('TimeTotalInHours', 0), errors='coerce') / 60,
                    'DistanceInMeters': data_to_update.get('DistanceInMeters', 0.0),
                    'CaloriesSpent': data_to_update.get('CaloriesSpent', 0.0),
                    'EstimatedActiveCal': data_to_update.get('estimated_calories', 0)
                }
                df = update_or_add_row(df, time_added, data_to_update.get('WorkoutType', ''), updates)

            elif from_where == 'input_activities':
                for activity, details in data_to_update.items():
                    updates = {
                        'HeartRateAverage': details['heart_rate'],
                        'TimeTotalInHours': details['duration'] / 60,
                        'DistanceInMeters': details['distance'],
                        'CaloriesSpent': details['calories_spent'],
                        'EstimatedActiveCal': details.get('estimated_calories', 0)
                    }
                    df = update_or_add_row(df, time_added, activity, updates)

            elif from_where == 'calories_consumed':
                if time_added in df.index:
                    df.loc[time_added, 'CaloriesConsumed'] += data_to_update
                else:
                    new_row = create_default_row(time_added)
                    new_row['CaloriesConsumed'] = data_to_update
                    df = pd.concat([df, new_row]).sort_index()

            FileSaver().save_raw_and_final_dataframes(df=df)  # Save updated DataFrame
            # Calculate TSS per discipline, TOTAL TSS and tss, atl, ctl, tsb
            w_df, tss_df, atl_df, ctl_df, tsb_df = calculate_total_tss_and_metrics_from_tss(df, 'update_final_df')

            FileSaver().save_tss_values_for_dashboard(tss_df, atl_df, ctl_df, tsb_df)

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

    def save_and_load_during_process(self, file_path=None, **kwargs):
        """
        Saves and Loads intermediate workout and/or activity data during the processing stage.

        Args:
            file_path (str, optional): Custom file path for loading the CSV files (default is None).
            **kwargs (DataFrame): Optional. One or both of workouts_tmp_df and/or activities_tmp_df.

        Returns:
            tuple: Loaded DataFrames for workouts_tmp_df and activities_tmp_df.
        """
        # Save the provided DataFrames
        FileSaver().save_during_process(file_path=file_path, **kwargs)

        if 'workouts_tmp_df' in kwargs:
            self.workouts_tmp_df = self._load_csv(file_path or self.file_path, 'workouts_tmp_df')
        if 'activities_tmp_df' in kwargs:
            self.activities_tmp_df = self._load_csv(file_path or self.file_path, 'activities_tmp_df')

    def load_user_nutrition(self, file_path=None): # NOTE: probably it should be called update, cuz it is loading and saving
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

        FileSaver().save_raw_and_final_dataframes(w_df = workouts_df, a_df=activities_df, foods_df=foods_df, file_path = 'data/raw/csv')
