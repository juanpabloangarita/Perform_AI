# Perform_AI.src.data_processing.py

import streamlit as st
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import csv
import os
import sys
import plotly.graph_objs as go
import plotly.io as pio

from src.calorie_calculations import calculate_total_calories
from src.calorie_estimation_models import estimate_calories_with_duration, estimate_calories_with_nixtla
from src.tss_calculations import calculate_total_tss_and_metrics_from_tss

from src.data_loader.files_extracting import FileLoader
from src.data_loader.files_saving import FileSaver
from params import CLOUD_ON, GIVEN_DATE, BEST_MODEL
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


def convert_to_datetime(df, date_col):
    """
    Convert specified column to datetime and set as index with uniform date format.

    Parameters:
        df (pd.DataFrame): DataFrame to process.
        date_col (str): Name of the date column.

    Returns:
        pd.DataFrame: DataFrame with 'Date' as index in datetime format (YYYY-MM-DD).
    """
    # Explicitly check for known date columns
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


def standardize_date_index(df):
    """
    Converts the index of the dataframe to datetime and formats it as 'YYYY-MM-DD'.

    Parameters:
    df (pd.DataFrame): DataFrame with a date index to be standardized.

    Returns:
    pd.DataFrame: DataFrame with the index formatted as 'YYYY-MM-DD'.
    """
    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    # Format index as 'YYYY-MM-DD'
    df.index = df.index.strftime('%Y-%m-%d')
    return df


def process_data(user_data, workouts=None):
    """Process and prepare data, estimate calories, and combine results."""
    sourcer = FileLoader()
    if CLOUD_ON == 'no':
        sourcer.load_initial_csv_files()

    tmp_workouts, activities_df = FileLoader().load_dfs(name_s=['workouts_df', 'activities_df'], file_path='data/raw/csv')
    workouts_df = workouts if workouts is not None else tmp_workouts

    workouts_df = clean_data_basic(workouts_df).copy()
    activities_df = clean_data_basic(activities_df).copy()

    workouts_df = convert_to_datetime(workouts_df, 'WorkoutDay').copy()
    activities_df = convert_to_datetime(activities_df, 'Date').copy()

    columns_to_keep_workouts = ['WorkoutType', 'Title', 'WorkoutDescription', 'CoachComments',
                                'HeartRateAverage', 'TimeTotalInHours', 'DistanceInMeters', 'PlannedDuration', 'PlannedDistanceInMeters']
    french_to_english = {
        'Type d\'activité': 'WorkoutType',
        'Titre': 'Title',
        'Fréquence cardiaque moyenne': 'HeartRateAverage',
        'Durée': 'TimeTotalInHours',
        'Distance': 'DistanceInMeters',
        'Calories': 'Calories'
    }
    columns_to_keep_activities = list(french_to_english.values())
    workouts_df = filter_and_translate_columns(workouts_df, {}, columns_to_keep_workouts).copy()
    activities_df = filter_and_translate_columns(activities_df, french_to_english, columns_to_keep_activities).copy()

    workouts_to_remove_both_dfs = ['Brick', 'Other', 'Strength', 'Day Off', 'HIIT', 'Exercice de respiration', 'Musculation']
    sports_types = {
        'Nat. piscine': 'Swim',
        'Cyclisme': 'Bike',
        'Course à pied': 'Run',
        "Vélo d'intérieur": 'Bike',
        'Cyclisme virtuel': 'Bike',
        'Course à pied sur tapis roulant': 'Run',
        'Natation': 'Swim',
    }
    workouts_df = filter_and_translate_workouts_column(workouts_df, workouts_to_remove_both_dfs).copy()
    activities_df = filter_and_translate_workouts_column(activities_df, workouts_to_remove_both_dfs, sports_types).copy()

    activities_df = activities_df.dropna()
    columns_to_float = ['HeartRateAverage', 'Calories', 'DistanceInMeters', 'TimeTotalInHours']
    activities_df = convert_data_types_for_activities(activities_df, columns_to_float).copy()
    workouts_df = filter_workouts_df_and_remove_nans(workouts_df).copy()

    w_df, tss_df, atl_df, ctl_df, tsb_df = calculate_total_tss_and_metrics_from_tss(workouts_df, 'data_processing')
    w_df_calories_calculated = calculate_total_calories(user_data, df=w_df)
    print_metrics_or_data('both', w_df_tmp=w_df, act_df_tmp=activities_df)

    # Model creation and predictions
    X_activities = activities_df.rename(columns={'TimeTotalInHours': 'TotalDuration'})
    y_activities = activities_df['Calories']
    w_df_calories_estimated, rmse_results = create_models_and_predict(X_activities, y_activities, w_df)
    print_metrics_or_data('rmse', rmse=rmse_results)
    print_metrics_or_data('both', w_df_tmp=w_df_calories_estimated, act_df_tmp=activities_df)

    # Nixtla forecast
    forecast_result = create_nixtla_and_predict(X_activities, y_activities, w_df)

    # Combine estimated and calculated calories
    w_df_calories_estimated.reset_index(inplace=True)
    w_df_calories_estimated['Date'] = pd.to_datetime(w_df_calories_estimated['Date'], format='%Y-%m-%d')
    w_df_calories_estimated = pd.merge(w_df_calories_estimated, forecast_result, on='Date', how='left')
    w_df_calories_estimated.set_index('Date', inplace=True)

    final_columns = ['WorkoutType', 'Title', 'WorkoutDescription', 'CoachComments', 'HeartRateAverage',
                     'TimeTotalInHours', 'DistanceInMeters', 'Run_Cal', 'Bike_Cal', 'Swim_Cal',
                     'TotalPassiveCal', 'CalculatedActiveCal', 'EstimatedActiveCal', 'AutoARIMA',
                     'AutoARIMA-lo-95', 'AutoARIMA-hi-95', 'Calories', 'CaloriesSpent', 'CaloriesConsumed']

    w_df_calories_estimated = standardize_date_index(w_df_calories_estimated)
    w_df_calories_calculated = standardize_date_index(w_df_calories_calculated)
    activities_df = standardize_date_index(activities_df)


    # Reset the index to ensure the 'date' is a column and not part of the index
    w_df_calories_estimated_reset = w_df_calories_estimated.reset_index()
    w_df_calories_calculated_reset = w_df_calories_calculated.reset_index()
    activities_df_reset = activities_df.reset_index()

    # Concatenate calculated and estimated calories
    w_df_combined = pd.concat([w_df_calories_estimated_reset, w_df_calories_calculated_reset], axis=1, join='inner')

    final_df = pd.concat([w_df_combined, activities_df_reset[['Date', 'Calories']]], axis=1)

    # Drop duplicate 'Date' columns if they exist
    final_df = final_df.loc[:,~final_df.columns.duplicated()]
    final_df = final_df.set_index('Date')
    final_df = final_df.reindex(columns=final_columns, fill_value=0.0)
    numeric_cols = final_df.select_dtypes(include=['float64', 'int64']).columns
    final_df[numeric_cols] = final_df[numeric_cols].fillna(0.0)
    final_df['ComplianceStatus'] = ''
    final_df['TSS'] = 0.0 # NOTE: probably this could be inserted as final columns, and used the reindex fill_value = 0.0 or a dictionary for this and before line


    return tss_df, atl_df, ctl_df, tsb_df, w_df_combined, activities_df, final_df
