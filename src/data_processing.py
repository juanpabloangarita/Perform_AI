# data_processing.py

# Import necessary libraries for data manipulation
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter

import csv
import os

import plotly.graph_objs as go
import plotly.io as pio

from params import *
from src.tss_calculations import * #WARNING WHY IT WORKED WITH .tss_calculations before

def load_csv(file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located - src
    dir_script_dir = os.path.dirname(script_dir) # Get the directory where the previous dir is located - PerformAI
    full_path = os.path.join(dir_script_dir, file_path)  # Construct the full path

    # Print the paths for debugging
    # print(f"Original file_path: {file_path}")
    # print(f"Script_dir: {dir_script_dir}")
    # print(f"Full_path: {full_path}")

    # # Original file_path: data/raw/csv/
    # # dir_script_dir: /Users/juanpabloangaritaafricano/code/juanpabloangarita/PerformAI
    # # Full_path: /Users/juanpabloangaritaafricano/code/juanpabloangarita/PerformAI/data/raw/csv/

    # Training Peaks -- Workout Files
    # from 03 of March to 03 of March next year
    workouts_2022_df = pd.read_csv(os.path.join(full_path, 'tp_workouts_2022-03-03_to_2023-03-03.csv'))
    workouts_2023_df = pd.read_csv(os.path.join(full_path, 'tp_workouts_2023-03-03_to_2024-03-03.csv'))
    # from 03 of March to 30 of August same year
    workouts_2024_df = pd.read_csv(os.path.join(full_path, 'tp_workouts_2024-03-03_to_2024-09-30.csv'))

    return workouts_2022_df, workouts_2023_df, workouts_2024_df


def clean_data(dfs, date_cols):
    # Threshold of % of NaN's per column we want to accept
    threshold = 0.5

    for df_name, df in dfs.items():

        #REMOVED DROP_HIGH_NA_COLUMNS CUZ WEIRD BEHAVIOUR, WILL NEED TO UPDATE CUZ NO NEED

        # Remove columns with high > threshold % NaN values
        # drop_high_na_columns(df, threshold) # WARNING - REMOVED THIS ONE
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


def process_data(workouts=None):

    if workouts is not None:
        workouts_df = workouts
    else:
        w_df1, w_df2, w_df3 = load_csv('data/raw/csv/') # WITHOUT THE / behind

        # Merge workouts DataFrames into one
        workouts_df = pd.concat([w_df1, w_df2, w_df3], ignore_index=True)

    dataframes = {
        #'activities': activities_df,
        #'sleep': sleep_df,
        #'health_metrics': health_metrics_df,
        'workouts': workouts_df
    }
    date_columns = {
        #'activities': 'Date', # as column
        #'sleep': 'Date', # as column
        #'health_metrics': 'Timestamp', # already as index
        'workouts': 'WorkoutDay' # as column
    }
    clean_data(dataframes, date_columns)

    columns_to_keep = ['Title', 'WorkoutType', 'HeartRateAverage', 'TimeTotalInHours', 'DistanceInMeters', 'PlannedDuration']
    dataframes['workouts'] = dataframes['workouts'][columns_to_keep].copy()

    # on a separate function than clean_data because different operations on workouts_df
    w_df = remove_no_training_days(dataframes['workouts'])

    # Calculate TSS per discipline and TOTAL TSS
    w_df = calculate_total_tss(w_df)

    # Calculate ATL, CTL, TSB from TSS
    tss_df, atl_df, ctl_df, tsb_df = calculate_metrics_from_tss(w_df)


    return tss_df, atl_df, ctl_df, tsb_df, w_df


# Add other data processing functions as needed
