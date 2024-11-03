# Perform_AI.src.data_helpers.py

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

from src.calorie_estimation_models import estimate_calories_with_duration, estimate_calories_with_nixtla
from src.data_loader.files_extracting import FileLoader
from params import GIVEN_DATE, BEST_MODEL
from typing import List


def clean_data_basic(df):
    """
    Clean data for the given dataframes.

    Parameters:
        dfs (dict): Dictionary of DataFrames to clean.
        date_cols (dict): Dictionary mapping DataFrame names to their date column names.
    """
    df = df.replace('--', np.nan)
    df = df.drop_duplicates()

    return df


def process_date_column(df, date_col=None, standardize=False):
    """
    Convert specified column to datetime, set as index, and standardize the date index if required.

    Parameters:
        df (pd.DataFrame): DataFrame to process.
        date_col (str, optional): Name of the date column. If provided, converts this column to datetime
                                   and sets it as the index. If None, only standardizes the date index.
        standardize (bool): If True, standardizes the date index format.

    Returns:
        pd.DataFrame: DataFrame with the date index formatted as 'YYYY-MM-DD' if standardize is True.
    """
    # If a date column is specified, convert it to datetime and set as index
    if date_col:
        if date_col in ['Date', 'WorkoutDay']:
            df['Date'] = pd.to_datetime(df[date_col])
            df = df.sort_values('Date')
            df = df.set_index('Date')
        elif date_col == 'Timestamp':
            df.index.name = 'Date'
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
        else:
            raise ValueError(f"Unrecognized date column: {date_col}")

        # Format the index to YYYY-MM-DD
        #df.index = df.index.date  # Keep only the date part
        df.index = df.index.normalize()  # Keep only the date part
        # today = datetime.today().date()
        # GIVEN_DATE = pd.to_datetime(today).normalize() # NOTE: to use had i left the index with the date format all along.

        # Drop the original date column if it exists
        if date_col in df.columns:
            df = df.drop(columns=date_col)

    elif standardize:
        # If no date column is provided but standardize is True, we are standardizing the index
        df.index = pd.to_datetime(df.index)
        df.index = df.index.strftime('%Y-%m-%d')
        df = df.reset_index()

    return df


def filter_and_translate_columns(df, column_mapping, columns_to_keep):
    """
    Translates column names in a DataFrame based on a given mapping and filters to keep only specified columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame with original column names.
    - column_mapping (dict): A dictionary mapping original column names to desired column names.
    - columns_to_keep (list, optional): List of columns to keep in the final DataFrame after renaming. Defaults to None.

    Returns:
    - pd.DataFrame: A DataFrame with renamed and filtered columns.
    """
    # Translate columns
    df_translated = df.rename(columns=column_mapping)

    # Determine columns to keep
    df_translated = df_translated[columns_to_keep]

    return df_translated


def filter_and_translate_workouts_column(df, workouts_to_remove, sports_mapping=None):
    """
    Filters and translates workout types in a DataFrame based on specified criteria.

    Parameters:
    - df (pd.DataFrame): The DataFrame with workout data.
    - workouts_to_remove (list): List of workout types to exclude from the DataFrame.
    - sports_mapping (dict, optional): A dictionary to translate workout types. Defaults to None.

    Returns:
    - pd.DataFrame: The DataFrame with filtered and translated workout types.
    """
    # Filter out unwanted workout types
    df_filtered = df[~df['WorkoutType'].isin(workouts_to_remove)].copy()

    # Apply mapping if provided
    if sports_mapping:
        df_filtered['WorkoutType'] = df_filtered['WorkoutType'].map(sports_mapping).fillna(df_filtered['WorkoutType'])

    return df_filtered


def convert_time_to_hours(time_str: str) -> float:
    """Convert a time string to hours."""
    try:
        return pd.to_timedelta(time_str).total_seconds() / 3600
    except Exception as e:
        print(f"Error converting time '{time_str}': {e}")
        return 0.0

def clean_calories(calories_str: str) -> float:
    """Remove thousands commas from calorie strings and convert to float."""
    try:
        return float(calories_str.replace(',', ''))
    except ValueError:
        print(f"Error cleaning calories '{calories_str}': cannot convert to float.")
        return 0.0

def convert_distance_to_meters(distance_str: str, workout_type: str) -> float:
    """Convert distance string to meters based on workout type."""
    try:
        distance_value = float(distance_str.replace(',', ''))  # Remove commas
        return distance_value if workout_type == 'Swim' else distance_value * 1000
    except ValueError:
        print(f"Error converting distance '{distance_str}' for workout '{workout_type}': cannot convert to float.")
        return 0.0

def convert_data_types_for_activities(df: pd.DataFrame, columns_to_modify: List[str]) -> pd.DataFrame:
    """Convert specified columns in a DataFrame to appropriate data types."""
    for col in columns_to_modify:
        if col == 'DistanceInMeters':
            df[col] = df.apply(lambda row: convert_distance_to_meters(row['DistanceInMeters'], row['WorkoutType']), axis=1)
        elif col in ['HeartRateAverage', 'Calories', 'TimeTotalInHours']:
            conversion_func = clean_calories if col == 'Calories' else (convert_time_to_hours if col == 'TimeTotalInHours' else float)
            df[col] = df[col].apply(conversion_func)

        df[col] = df[col].astype('float64')  # Ensure the column is in float64 format

    return df


def filter_workouts_df_and_remove_nans(df, given_date = GIVEN_DATE):

    before_df = df[df.index < given_date].copy()
    after_df = df[df.index >= given_date].copy()
    # Remove rows, before the given date, where i didn't train, meaning, where HR and Total Time is nan.
    before_df_cleaned = before_df[~(before_df['HeartRateAverage'].isna() & before_df['TimeTotalInHours'].isna())].copy() # NOTE: HERE IS THE PART THAT CAUSES THE WEIRD BEHAVIOUR. Explanation below
    # TODO: (BTW, I DON'T NEED TO REMOVE THE HEARTRATEAVERAGE.ISNA, since what's important for me is timetotalinhours only)

    # Remove rows, after the given date, where Planned Duration is nan, which means there is no info on training, so no tss
    after_df = after_df[after_df['PlannedDuration'].notna()]

    # Concatenate before and after dataframes
    w_df = pd.concat([before_df_cleaned, after_df])

    object_cols = w_df.select_dtypes(include=['object']).columns
    w_df[object_cols] = w_df[object_cols].fillna('')

    return w_df


def print_metrics_or_data(keyword, rmse=None, w_df_tmp=None, act_df_tmp=None):
    """
    Print performance metrics or display DataFrames for activities and/or workouts based on the keyword.

    Args:
        keyword (str): Specify what to print ('rmse', 'activities', 'workouts', or 'both').
        rmse (list, optional): List of RMSE results to print if keyword is 'rmse'.
        w_df_tmp (DataFrame, optional): DataFrame for workouts to be saved and printed.
        act_df_tmp (DataFrame, optional): DataFrame for activities to be saved and printed.
    """
    # Create a FileLoader instance and load the temporary DataFrames
    loader = FileLoader()
    loader.save_and_load_during_process(workouts_tmp_df=w_df_tmp, activities_tmp_df=act_df_tmp)
    tmp_workouts = loader.workouts_tmp_df
    tmp_activities = loader.activities_tmp_df

    if keyword == "rmse" and rmse is not None:
        print("\nPerformance Metrics:\n")
        for result in rmse:
            print(f"{result['name']} RMSE: {result['rmse']}")
        print("\n")

    elif keyword in {"activities", "workouts", "both"}:
        if keyword in {"activities", "both"} and tmp_activities is not None:
            print("\nActivities DataFrame:\n", tmp_activities.head(), "\n")
        if keyword in {"workouts", "both"} and tmp_workouts is not None:
            print("\nWorkouts DataFrame:\n", tmp_workouts.head(), "\n")
    else:
        print("No valid data provided for the specified keyword.")


def create_models_and_predict(X_act, y_act, total_workouts):
    """Create models for calorie estimation and predict on total workouts."""
    rmse_results = estimate_calories_with_duration(X_act, y_act)

    # Load preprocessing pipeline and linear model
    preprocessing_pipeline = FileLoader().load_models('preprocessing_pipeline')
    linear_model = FileLoader().load_models(BEST_MODEL)

    # Fill missing values in 'TimeTotalInHours' and 'DistanceInMeters'
    total_workouts = total_workouts.fillna({
        'TimeTotalInHours': total_workouts['PlannedDuration'],
        'DistanceInMeters': total_workouts['PlannedDistanceInMeters']
    })
    total_workouts['TotalDuration'] = total_workouts['TimeTotalInHours']

    # Filter rows with complete data for transformation
    mask_total = total_workouts['TotalDuration'].notna()
    total_workouts_transformed = preprocessing_pipeline.transform(total_workouts[mask_total])

    # Estimate calories
    total_workouts.loc[mask_total, 'EstimatedActiveCal'] = linear_model.predict(total_workouts_transformed)

    return total_workouts, rmse_results


def create_nixtla_and_predict(X_activities_df, y_activities_df, w_df):
    """Generate Nixtla predictions for future workouts."""
    future_workouts_df = w_df[w_df.index >= GIVEN_DATE].copy()
    future_workouts_df = future_workouts_df.rename(columns={'PlannedDuration': 'TotalDuration'})

    forecast, sf = estimate_calories_with_nixtla(X_activities_df, y_activities_df, future_workouts_df)
    forecast = forecast.rename(columns={'ds': 'Date'})
    forecast['Date'] = pd.to_datetime(forecast['Date'], format='%Y-%m-%d')

    return forecast
