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
from src.algorithms import *

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
    workouts_2024_df = pd.read_csv(os.path.join(full_path, 'tp_workouts_2024-03-03_to_2024-09-30.csv'))

    # ACTIVITIES GARMIN
    # Garmin files
    # From March 12 of 2022 to July 14 2024
    activities_df = pd.read_csv(os.path.join(full_path,'activities.csv'))

    return workouts_2022_df, workouts_2023_df, workouts_2024_df, activities_df


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


def clean_activities(df):
    columns_to_keep_activities = ["Type d'activité", 'Distance', 'Calories', 'Durée', 'Fréquence cardiaque moyenne']
    df = df[columns_to_keep_activities].copy()

    df = df[df['Fréquence cardiaque moyenne'].notna()].copy()

    sports_types = {
    'Nat. piscine': 'Swim',
    'Cyclisme': 'Bike',
    'Course à pied': 'Run',
    "Vélo d'intérieur": 'Bike',
    'Cyclisme virtuel': 'Bike',
    'Course à pied sur tapis roulant': 'Run',
    'Natation': 'Swim',
}
    df = df[(df["Type d'activité"] != 'HIIT') & (df["Type d'activité"] != 'Exercice de respiration') & (df["Type d'activité"] != 'Musculation')].copy()
    df["Type d'activité"] = df["Type d'activité"].apply(lambda x: sports_types[x])

    # Convert Durée from 'hh:mm:ss' to total minutes
    df['Durée'] = pd.to_timedelta(df['Durée']).dt.total_seconds() / 60  # Convert to minutes

    # Convert relevant columns to numeric (remove commas, etc.)
    df['Distance'] = pd.to_numeric(df['Distance'].str.replace(',', '.'), errors='coerce')
    df['Calories'] = pd.to_numeric(df['Calories'], errors='coerce')

    # Drop rows with NaN values in critical columns
    df = df.dropna(subset=['Distance', 'Calories', 'Durée', 'Fréquence cardiaque moyenne'])

    return df


def process_data(workouts=None):
    ## -
    # if workouts is not None:
    #     workouts_df = workouts
    # else:
    #     w_df1, w_df2, w_df3, activities_df = load_csv('data/raw/csv/') # WITHOUT THE / behind

    #     # Merge workouts DataFrames into one
    #     workouts_df = pd.concat([w_df1, w_df2, w_df3], ignore_index=True)
    ## -

    ## --
    w_df1, w_df2, w_df3, activities_df = load_csv('data/raw/csv/') # WITHOUT THE / behind

    # Merge workouts DataFrames into one
    workouts_df = pd.concat([w_df1, w_df2, w_df3], ignore_index=True)

    if workouts is not None:
        workouts_df = workouts
    ## --

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
    columns_to_keep_workouts = ['Title', 'WorkoutType', 'HeartRateAverage', 'TimeTotalInHours', 'DistanceInMeters', 'PlannedDuration']
    dataframes['workouts'] = dataframes['workouts'][columns_to_keep_workouts].copy()

    # on a separate function than clean_data because different operations on workouts_df
    w_df = remove_no_training_days(dataframes['workouts'])

    # Calculate TSS per discipline and TOTAL TSS
    w_df = calculate_total_tss(w_df)

    # Calculate Total Calories from TSS
    w_df_calories = calculate_total_calories(df=w_df) #, weight, height, age, gender, vo2_max, resting_hr) # WARNING, WHY WITHOUT THIS?

    # Calculate ATL, CTL, TSB from TSS calories
    tss_df, atl_df, ctl_df, tsb_df = calculate_metrics_from_tss(w_df_calories)



    # ACTIVITIES
    activities_df = clean_activities(dataframes['activities_df'])

    # Separate past and future workouts
    past_workouts_df = w_df_calories[w_df_calories['Date'] <= GIVEN_DATE]
    future_workouts_df = w_df_calories[w_df_calories['Date'] > GIVEN_DATE]



    #return w_df_calories
    return tss_df, atl_df, ctl_df, tsb_df, w_df_calories

    # # Calculate ATL, CTL, TSB from TSS
    # tss_df, atl_df, ctl_df, tsb_df = calculate_metrics_from_tss(w_df)

    # return tss_df, atl_df, ctl_df, tsb_df, w_df


# Add other data processing functions as needed
