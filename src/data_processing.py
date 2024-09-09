# data_processing.py
import streamlit as st
# Import necessary libraries for data manipulation
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

import csv
import os

import plotly.graph_objs as go
import plotly.io as pio

from params import *
from src.calorie_calculations import calculate_total_calories
from src.tss_calculations import * #WARNING WHY IT WORKED WITH .tss_calculations before
from src.calorie_calculations import *
from src.calorie_estimation_models import *
from src.calorie_estimation_models import estimate_calories_with_workout_type, estimate_calories_without_workout_type

def load_csv(file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located - src
    dir_script_dir = os.path.dirname(script_dir) # Get the directory where the previous dir is located - PerformAI
    full_path = os.path.join(dir_script_dir, file_path)  # Construct the full path

    # Debugging output displayed on Streamlit UI
    # st.write(f"Script directory: {script_dir}")
    # st.write(f"Parent directory (PerformAI): {dir_script_dir}")
    # st.write(f"Constructed full path: {full_path}")

    # if not os.path.exists(full_path):
    #     st.write(f"Path does not exist: {full_path}")
    # else:
    #     st.write(f"Path exists: {full_path}")

    # Training Peaks -- Workout Files
    # from 03 of March to 03 of March next year
    workouts_2022_df = pd.read_csv(os.path.join(full_path, 'tp_workouts_2022-03-03_to_2023-03-03.csv'))
    workouts_2023_df = pd.read_csv(os.path.join(full_path, 'tp_workouts_2023-03-03_to_2024-03-03.csv'))
    # from 03 of March to 30 of August same year
    workouts_2024_df = pd.read_csv(os.path.join(full_path, 'tp_workouts_2024-03-03_to_2024-11-24.csv'))

    # ACTIVITIES GARMIN
    # Garmin files REAL CALORIES
    # From March 12 of 2022 to July 14 2024
    activities_df_all_years = pd.read_csv(os.path.join(full_path,'activities.csv'))

    return workouts_2022_df, workouts_2023_df, workouts_2024_df, activities_df_all_years


def save_csv(file_path, w_df, a_df, df):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located - src
    dir_script_dir = os.path.dirname(script_dir) # Get the directory where the previous dir is located - PerformAI
    full_path = os.path.join(dir_script_dir, file_path)  # Construct the full path

    # save workouts_df
    w_df.to_csv(os.path.join(full_path, 'workouts_df.csv'))
    # save activities_df
    a_df.to_csv(os.path.join(full_path, 'activities_df.csv'))
    # save final_df
    df.to_csv(os.path.join(full_path, 'final_df.csv'))


def clean_data(dfs, date_cols):
    # Threshold of % of NaN's per column we want to accept
    threshold = 0.5

    for df_name, df in dfs.items():

        #REMOVED DROP_HIGH_NA_COLUMNS CUZ WEIRD BEHAVIOUR, WILL NEED TO UPDATE CUZ NO NEED

        # Remove columns with high > threshold % NaN values
        # drop_high_na_columns(df, threshold) # WARNING - REMOVED THIS ONE

        # Replace '--' with NaN
        df.replace('--', np.nan, inplace=True)
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        # Change date format & place it as index
        if df_name == 'sleep':
            continue
        date_col = date_cols[df_name]
        convert_to_datetime(df, date_col)


def drop_high_na_columns(df, threshold):
    # Calculate the percentage of missing values for each column
    na_percentage = df.isna().sum() / len(df)
    # Select columns to drop
    columns_to_drop = na_percentage[na_percentage > threshold].index
    # Drop the columns
    df.drop(columns=columns_to_drop, inplace=True)
    #return columns_to_drop


def convert_to_datetime(df, date_col):
    if date_col != 'Timestamp':
        df['Date'] = pd.to_datetime(df[date_col])
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)
    else:
        df.index.name = 'Date'
        df.sort_values('Date', inplace=True)
        df.set_index(pd.to_datetime(df.index), inplace=True)

    #df.sort_values('Date', inplace=True)
    # Remove the date column to normalize all date columns with the same name
    if date_col in df.columns:
        # This removes the column name date_col
        df.drop(columns=date_col, inplace=True)
    #return df

def remove_no_training_days(df, given_date = GIVEN_DATE):
    before_df = df[df.index < given_date].copy()
    after_df = df[df.index >= given_date].copy()
    # Remove rows, before the given date, where i didn't train, meaning, where HR and Total Time is nan.
    before_df_cleaned = before_df[~(before_df['HeartRateAverage'].isna() & before_df['TimeTotalInHours'].isna())].copy()

    # Remove rows, after the given date, where Planned Duration is nan, which means there is no info on training, so no tss
    after_df = after_df[after_df['PlannedDuration'].notna()]

    # Concatenate before and after dataframes
    w_df = pd.concat([before_df_cleaned, after_df])
    # Keep dates where there was a Run Swim or Bike training Plan
    w_df = w_df[(w_df['WorkoutType'] == 'Run') | (w_df['WorkoutType'] == 'Swim') | (w_df['WorkoutType'] == 'Bike')].copy()

    return w_df


def clean_activities(df):
    columns_to_keep_activities = ["Type d'activité", 'Distance', 'Calories', 'Durée', 'Fréquence cardiaque moyenne']
    df = df[columns_to_keep_activities].copy()
    df = df.rename(columns={
        'Distance': 'DistanceInMeters',
        'Durée': 'TimeTotalInHours',
        'Fréquence cardiaque moyenne': 'HeartRateAverage',
        'Type d\'activité': 'WorkoutType'
    })
    df['HeartRateAverage'] = pd.to_numeric(df['HeartRateAverage'])

    df = df[df['HeartRateAverage'].notna()].copy()

    df = df[(df["WorkoutType"] != 'HIIT') & (df["WorkoutType"] != 'Exercice de respiration') & (df["WorkoutType"] != 'Musculation')].copy()

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
    df['TimeTotalInHours'] = pd.to_timedelta(df['TimeTotalInHours']).dt.total_seconds() / 60  # Convert to minutes

    # Convert relevant columns to numeric (remove commas, etc.)
    df['DistanceInMeters'] = pd.to_numeric(df['DistanceInMeters'].str.replace(',', '.'), errors='coerce')
    df['Calories'] = pd.to_numeric(df['Calories'], errors='coerce')

    # Drop rows with NaN values in critical columns
    df = df.dropna(subset=['DistanceInMeters', 'Calories', 'TimeTotalInHours', 'HeartRateAverage'])

    return df

def print_performances(models):
    # Printing the performance metrics
    print("Performance Metrics:")
    print("\nWith Heart Rate:")
    print(f"Linear Regression RMSE: {models['rmse_lr_y_hr']}")
    print(f"Random Forest RMSE: {models['rmse_rf_y_hr']}")
    print(f"Gradient Boosting RMSE: {models['rmse_gb_y_hr']}")
    print(f"LIGHTGBM RMSE: {models['rmse_lgb_y_hr']}")
    print(f"XGBOOST RMSE: {models['rmse_xgb_y_hr']}")

    print("\nWithout Heart Rate:")
    print(f"Linear Regression RMSE: {models['rmse_lr_no_hr']}")
    print(f"Random Forest RMSE: {models['rmse_rf_no_hr']}")
    print(f"Gradient Boosting RMSE: {models['rmse_gb_no_hr']}")
    print(f"LIGHTGBM RMSE: {models['rmse_lgb_no_hr']}")
    print(f"XGBOOST RMSE: {models['rmse_xgb_no_hr']}")


def aggregate_by_date(cal_estimated_df, cal_calculated_df, activities):
    cal_estimated_df.index = pd.to_datetime(cal_estimated_df.index).normalize()
    cal_calculated_df.index = pd.to_datetime(cal_calculated_df.index).normalize()
    activities.index = pd.to_datetime(activities.index).normalize()# HAD IT BEEN A NORMAL COLUMN - meaning 'Date' not as index, WE WOULD HAVE NEEDED TO DO THE FOLLOWING
    # activities['Date'] = pd.to_datetime(activities['Date']).dt.normalize()

    # aggregate by date
    cal_estimated_df_agg = cal_estimated_df.groupby('Date').agg('sum')
    cal_calculated_df_agg = cal_calculated_df.groupby('Date').agg('sum')
    activities_agg = activities.groupby('Date').agg('sum')

    return cal_estimated_df_agg, cal_calculated_df_agg, activities_agg


def process_data(workouts=None):
    w_df1, w_df2, w_df3, activities_df = load_csv('data/raw/csv/') # WITHOUT THE / behind

    # Merge workouts DataFrames into one
    workouts_df = pd.concat([w_df1, w_df2, w_df3], ignore_index=True)

    if workouts is not None:
        workouts_df = workouts #WARNING

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
    clean_data(dataframes, date_columns)

    # WORKOUTS
    columns_to_keep_workouts = ['Title', 'WorkoutType', 'HeartRateAverage', 'TimeTotalInHours', 'DistanceInMeters', 'PlannedDuration', 'PlannedDistanceInMeters']
    dataframes['workouts'] = dataframes['workouts'][columns_to_keep_workouts].copy()

    # on a separate function than clean_data because different operations on workouts_df
    w_df = remove_no_training_days(dataframes['workouts'])

    # Calculate TSS per discipline and TOTAL TSS
    w_df = calculate_total_tss(w_df)

    # # Calculate ATL, CTL, TSB from TSS
    tss_df, atl_df, ctl_df, tsb_df = calculate_metrics_from_tss(w_df)

    # ACTIVITIES
    activities_df = clean_activities(dataframes['activities'])

    # Separate past and future workouts
    past_workouts_df = w_df.loc[w_df.index < GIVEN_DATE]
    future_workouts_df = w_df.loc[w_df.index >= GIVEN_DATE]

    workout_type = True
    if workout_type:
        # worse performance of models
        w_df_calories_estimated, models_dict = estimate_calories_with_workout_type(activities_df, past_workouts_df, future_workouts_df)
        print(models_dict)
    else:
        # CURRENTLY WORKING WITH THIS ONE
        # better performance of models
        w_df_calories_estimated, models_dict= estimate_calories_without_workout_type(activities_df, past_workouts_df, future_workouts_df)
        print(models_dict)

    # print_performances(models_dict)

    # Calculate Total Calories from TSS
    w_df_calories_calculated = calculate_total_calories(df=w_df) #, weight, height, age, gender, vo2_max, resting_hr) # WARNING, WHY WITHOUT THIS?

    w_df_cal_est, w_df_cal_calc, activities_df = aggregate_by_date(w_df_calories_estimated, w_df_calories_calculated, activities_df)
    w_df_calories_estimated_plus_calculated = pd.concat([w_df_cal_est, w_df_cal_calc], axis=1, join='inner')

    final_columns = ['WorkoutType', 'HeartRateAverage', 'TimeTotalInHours', 'DistanceInMeters', 'PlannedDuration', 'PlannedDistanceInMeters', 'TotalPassiveCalories', 'EstimatedCalories']
    final_df = pd.concat([w_df_calories_estimated_plus_calculated[final_columns], activities_df['Calories']], axis=1)

    save_csv('data/processed/csv/', w_df_calories_estimated_plus_calculated, activities_df, final_df)

    return tss_df, atl_df, ctl_df, tsb_df, w_df_calories_estimated_plus_calculated, activities_df, final_df


# Add other data processing functions as needed
