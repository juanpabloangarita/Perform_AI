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


def clean_data_basic(dfs, date_cols):
    """
    Clean data for the given dataframes.

    Parameters:
        dfs (dict): Dictionary of DataFrames to clean.
        date_cols (dict): Dictionary mapping DataFrame names to their date column names.
    """
    for df_name, df in dfs.items():
        df.replace('--', np.nan, inplace=True)
        df.drop_duplicates(inplace=True)

        if df_name == 'sleep':
            continue
        convert_to_datetime(df, date_cols[df_name])


def convert_to_datetime(df, date_col):
    """
    Convert specified column to datetime and set as index.

    Parameters:
        df (pd.DataFrame): DataFrame to process.
        date_col (str): Name of the date column.
    """
    if date_col != 'Timestamp':
        df['Date'] = pd.to_datetime(df[date_col])
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)
    else:
        df.index.name = 'Date'
        df.sort_values('Date', inplace=True)
        df.set_index(pd.to_datetime(df.index), inplace=True)

    if date_col in df.columns:
        df.drop(columns=date_col, inplace=True)


def clean_activities(df):
    """
    Clean activity data to keep relevant columns and rename them.

    Parameters:
        df (pd.DataFrame): DataFrame containing activity data.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    columns_to_keep = ["Type d'activité", 'Distance', 'Calories', 'Durée', 'Fréquence cardiaque moyenne']
    df = df[columns_to_keep].copy().rename(columns={
        'Distance': 'DistanceInMeters',
        'Durée': 'TimeTotalInHours',
        'Fréquence cardiaque moyenne': 'HeartRateAverage',
        'Type d\'activité': 'WorkoutType'
    })

    df['HeartRateAverage'] = pd.to_numeric(df['HeartRateAverage'], errors='coerce')
    df = df[df['HeartRateAverage'].notna()]

    df = df[~df["WorkoutType"].isin(['HIIT', 'Exercice de respiration', 'Musculation'])].copy()

    sports_types = {
        'Nat. piscine': 'Swim',
        'Cyclisme': 'Bike',
        'Course à pied': 'Run',
        "Vélo d'intérieur": 'Bike',
        'Cyclisme virtuel': 'Bike',
        'Course à pied sur tapis roulant': 'Run',
        'Natation': 'Swim',
    }
    df["WorkoutType"] = df["WorkoutType"].apply(lambda x: sports_types[x])

    # Convert Durée from 'hh:mm:ss' to total minutes
    df['TimeTotalInHours'] = pd.to_timedelta(df['TimeTotalInHours']).dt.total_seconds() / 3600  # Convert to Hours

    # Convert relevant columns to numeric (remove commas, etc.)
    df['DistanceInMeters'] = pd.to_numeric(df['DistanceInMeters'].str.replace(',', '.'), errors='coerce')
    df['Calories'] = pd.to_numeric(df['Calories'], errors='coerce')

    # Drop rows with NaN values in critical columns
    df = df.dropna(subset=['DistanceInMeters', 'Calories', 'TimeTotalInHours', 'HeartRateAverage'])

    # df = df[df['DistanceInMeters']>0].copy() # NOTE: not needed since, i will be using only TotalDuration or TimeTotalInHours

    return df


def filter_workouts_and_remove_nans(df, given_date = GIVEN_DATE):
    columns_to_keep_workouts = ['WorkoutType', 'Title', 'WorkoutDescription', 'CoachComments', 'HeartRateAverage', 'TimeTotalInHours', 'DistanceInMeters', 'PlannedDuration', 'PlannedDistanceInMeters']
    df = df[columns_to_keep_workouts].copy()

    before_df = df[df.index < given_date].copy()
    after_df = df[df.index >= given_date].copy()
    # Remove rows, before the given date, where i didn't train, meaning, where HR and Total Time is nan.
    before_df_cleaned = before_df[~(before_df['HeartRateAverage'].isna() & before_df['TimeTotalInHours'].isna())].copy() # NOTE: HERE IS THE PART THAT CAUSES THE WEIRD BEHAVIOUR. Explanation below
    # TODO: (BTW, I DON'T NEED TO REMOVE THE HEARTRATEAVERAGE.ISNA, since what's important for me is timetotalinhours only)

    # Remove rows, after the given date, where Planned Duration is nan, which means there is no info on training, so no tss
    after_df = after_df[after_df['PlannedDuration'].notna()]

    # Concatenate before and after dataframes
    w_df = pd.concat([before_df_cleaned, after_df])
    # Keep dates where there was a Run Swim or Bike training Plan
    w_df = w_df[w_df['WorkoutType'].isin(['Run', 'Swim', 'Bike'])]

    # Fill NaN values in object columns with an empty string
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
    # Save temporary DataFrames during processing NOTE: the following line is an alternative to save only to the FileLoader
    # FileSaver().save_during_process(workouts_tmp_df=w_df_tmp, activities_tmp_df=act_df_tmp)

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

    sourcer.load_raw_and_final_dataframes('data/raw/csv')
    activities_df = sourcer.activities_raw
    workouts_df = workouts if workouts is not None else sourcer.workouts_raw

    dataframes = {
        'activities': activities_df,
        #'sleep': sleep_df,
        #'health_metrics': health_metrics_df,
        'workouts': workouts_df
    }
    date_columns = {
        'activities': 'Date', # as column
        #'sleep': 'Date', # as column
        #'health_metrics': 'Timestamp', # already as index
        'workouts': 'WorkoutDay' # as column
    }
    clean_data_basic(dataframes, date_columns)

    w_df = filter_workouts_and_remove_nans(dataframes['workouts'])
    w_df, tss_df, atl_df, ctl_df, tsb_df = calculate_total_tss_and_metrics_from_tss(w_df, 'data_processing')
    w_df_calories_calculated = calculate_total_calories(user_data, df=w_df)

    activities_df = clean_activities(dataframes['activities'])
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

    final_columns = ['WorkoutType', 'Title', 'WorkoutDescription', 'CoachComments', 'HeartRateAverage', 'TimeTotalInHours',
                     'DistanceInMeters', 'PlannedDuration', 'PlannedDistanceInMeters', 'Run_Cal', 'Bike_Cal', 'Swim_Cal',
                     'TotalPassiveCal', 'CalculatedActiveCal', 'EstimatedActiveCal', 'AutoARIMA',  'AutoARIMA-lo-95',  'AutoARIMA-hi-95', 'Calories', 'CaloriesSpent', 'CaloriesConsumed']

    w_df_calories_estimated = standardize_date_index(w_df_calories_estimated)
    w_df_calories_calculated = standardize_date_index(w_df_calories_calculated)
    activities_df = standardize_date_index(activities_df)


    # Reset the index to ensure the 'date' is a column and not part of the index
    w_df_calories_estimated_reset = w_df_calories_estimated.reset_index()
    w_df_calories_calculated_reset = w_df_calories_calculated.reset_index()
    activities_df_reset = activities_df.reset_index()

    # # Concatenate the two dataframes, now using columns and not index
    w_df_calories_estimated_plus_calculated = pd.concat([w_df_calories_estimated_reset, w_df_calories_calculated_reset], axis=1, join='inner')

    final_df = pd.concat([w_df_calories_estimated_plus_calculated, activities_df_reset[['Date', 'Calories']]], axis=1)

    # Drop duplicate 'Date' columns if they exist
    final_df = final_df.loc[:,~final_df.columns.duplicated()]

    final_df = final_df.set_index('Date')

    final_df = final_df.reindex(columns=final_columns, fill_value=0.0)

    numeric_cols = final_df.select_dtypes(include=['float64', 'int64']).columns
    final_df[numeric_cols] = final_df[numeric_cols].fillna(0.0)

    final_df = final_df.drop(columns=['PlannedDuration', 'PlannedDistanceInMeters'])
    final_df['ComplianceStatus'] = ''
    final_df['TSS'] = 0.0 # NOTE: probably this could be inserted as final columns, and used the reindex fill_value = 0.0 or a dictionary for this and before line


    return tss_df, atl_df, ctl_df, tsb_df, w_df_calories_estimated_plus_calculated, activities_df, final_df
