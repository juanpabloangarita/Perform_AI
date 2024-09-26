# data_processing.py
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

from params import *
from src.calorie_calculations import calculate_total_calories
from src.tss_calculations import * # NOTE: WHY IT WORKED WITH .tss_calculations before
from src.calorie_calculations import *
from src.calorie_estimation_models import *
from src.calorie_estimation_models import estimate_calories, estimate_calories_with_duration


def get_full_path(file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
    dir_script_dir = os.path.dirname(script_dir)  # Get the directory where the previous dir is located
    full_path = os.path.join(dir_script_dir, file_path)  # Construct the full path
    return full_path


def load_csv(file_path):
    full_path = get_full_path(file_path)
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


def save_final_csv(file_path, w_df, a_df, df):
    full_path = get_full_path(file_path)
    w_df.to_csv(os.path.join(full_path, 'workouts_df.csv'), na_rep='')
    a_df.to_csv(os.path.join(full_path, 'activities_df.csv'), na_rep='')
    df.to_csv(os.path.join(full_path, 'final_df.csv'), index=True, na_rep='')


def load_and_update_final_csv(file_path, from_where, time_added=None, data_to_update=None):
    full_path = get_full_path(file_path)
    df = pd.read_csv(os.path.join(full_path, 'final_df.csv'), index_col=0, na_filter=False)
    if from_where == 'home':
        return df
    elif from_where == 'training_peaks':
        for activity_dict in data_to_update:
            # Extracting values from the activity_dict
            date_str = activity_dict.get('Date', '')
            compliance_status = activity_dict.get('compliance_status', '')
            workout_type = activity_dict.get('WorkoutType', '')  # 'Bike', 'Run', 'Swim'
            title = activity_dict.get('Title', '')
            description = activity_dict.get('WorkoutDescription', '')
            coach_comments = activity_dict.get('CoachComments', '')
            duration = activity_dict.get('duration', 0.0)  # Duration in minutes # FIXME: TRANSFORM CALCULATE
            tss = float(activity_dict.get('tss', 0.0))  # TSS value

            date_str = pd.to_datetime(date_str)
            date_str = date_str.strftime('%Y-%m-%d')

            # Calculate duration in hours
            duration_hours = pd.to_timedelta(duration).total_seconds() / 3600

            existing_row_mask = (df.index == date_str) & (df['WorkoutType'] == workout_type)
            existing_row = df[existing_row_mask]

            if not existing_row.empty:
                # Update the specific row(s) with the correct column-value pairs
                df.loc[existing_row_mask, ['WorkoutType', 'Title',
                                        'WorkoutDescription', 'CoachComments', 'TimeTotalInHours']] = [
                    workout_type,
                    title,
                    description,
                    coach_comments,
                    duration_hours
                ]   # compliance_status, # tss
            else:
                # If the 'Date' is new, add a new row
                df.loc[date_str] = {
                    'Date': date_str,
                    #'compliance_status': compliance_status,
                    'WorkoutType': workout_type,
                    'Title': title,
                    'WorkoutDescription': description,
                    'CoachComments': coach_comments,
                    'TimeTotalInHours': duration_hours,
                    'DistanceInMeters': 0.0,
                    'CaloriesSpent': 0.0
                }
            try:
                df.to_csv(os.path.join(full_path, 'final_df.csv'), index=True, mode='w', na_rep='')
                sys.stdout.flush()
                print("File saved successfully")
            except Exception as e:
                print(f"Error saving final_df: {e}")

    elif from_where == 'plan_my_week':
        workout_type = data_to_update.get('WorkoutType', '')
        duration_hours = data_to_update.get('TimeTotalInHours', 0.0)
        distance = data_to_update.get('DistanceInMeters', 0.0)
        calories_spent = data_to_update.get('CaloriesSpent', 0.0)


        # Filter rows for the specific workout type on the given date
        existing_row_mask = (df.index == time_added) & (df['WorkoutType'] == workout_type)
        existing_row = df[existing_row_mask]

        if not existing_row.empty:
            df.loc[existing_row_mask, 'TimeTotalInHours'] = pd.to_numeric(duration_hours, errors='coerce')
            df.loc[existing_row_mask, 'DistanceInMeters'] = distance
            df.loc[existing_row_mask, 'CaloriesSpent'] = calories_spent
            try:
                df.to_csv(os.path.join(full_path, 'final_df.csv'), index=True, mode='w', na_rep='')
                sys.stdout.flush()
                print("File saved successfully")
            except Exception as e:
                print(f"Error saving final_df: {e}")
    else:
        time_added = pd.to_datetime(time_added) # NOTE: IT SEEMS REDUNDANT, CUZ before sending it i am already doing this
        time_added = time_added.strftime('%Y-%m-%d') # NOTE: IT SEEMS REDUNDANT, CUZ before sending it i am already doing this

        if from_where == "input_activities":
            # Loop through each activity and update the relevant columns
            for activity, details in data_to_update.items():
                workout_type = activity  # Corresponds to 'Bike', 'Run', 'Swim'
                heart_rate = details['heart_rate']
                duration_hours = details['duration'] / 60  # Convert duration from minutes to hours
                distance = details['distance']
                calories_spent = details['calories_spent']

                # Filter rows for the specific workout type on the given date
                existing_row_mask = (df.index == time_added) & (df['WorkoutType'] == workout_type)
                existing_row = df[existing_row_mask]

                if not existing_row.empty:
                    # Update the existing row for the specified WorkoutType
                    df.loc[existing_row_mask, 'HeartRateAverage'] += heart_rate  # Accumulate heart rates if needed
                    df.loc[existing_row_mask, 'TimeTotalInHours'] += pd.to_numeric(duration_hours, errors='coerce')
                    df.loc[existing_row_mask, 'DistanceInMeters'] += distance
                    df.loc[existing_row_mask, 'CaloriesSpent'] += calories_spent
                else:
                    new_row = pd.DataFrame({
                        'WorkoutType': [workout_type],
                        'Title': [''],  # Set default or placeholder for other columns
                        'WorkoutDescription': [''],
                        'CoachComments': [''],
                        'HeartRateAverage': [heart_rate],
                        'TimeTotalInHours': [duration_hours],
                        'DistanceInMeters': [distance],
                        'Run_Cal': [0.0],
                        'Bike_Cal': [0.0],
                        'Swim_Cal': [0.0],
                        'TotalPassiveCal': [0.0],
                        'CalculatedActiveCal': [0.0],
                        'EstimatedActiveCal': [0.0],
                        'Calories': [0.0],
                        'CaloriesSpent': [calories_spent],
                        'CaloriesConsumed': [0.0]  # Set default values for all columns
                    }, index=[time_added])

                    df = pd.concat([df, new_row])
                    df = df.sort_index()


        elif from_where == "calories_consumed":
            # Update the CaloriesConsumed column
            if time_added in df.index:
                df.loc[time_added, 'CaloriesConsumed'] += data_to_update  # Add new calories consumed
            else:
                new_row = pd.DataFrame({
                    'WorkoutType': [''],  # Keep as an empty string for WorkoutType
                    'Title': [''],  # Empty string for Title
                    'WorkoutDescription': [''],  # Empty string for WorkoutDescription
                    'CoachComments': [''],  # Empty string for CoachComments
                    'HeartRateAverage': [0.0],  # Default value for numeric columns
                    'TimeTotalInHours': [0.0],
                    'DistanceInMeters': [0.0],
                    'Run_Cal': [0.0],  # Default values for calories
                    'Bike_Cal': [0.0],
                    'Swim_Cal': [0.0],
                    'TotalPassiveCal': [0.0],
                    'CalculatedActiveCal': [0.0],
                    'EstimatedActiveCal': [0.0],
                    'Calories': [0.0],
                    'CaloriesSpent': [0.0],
                    'CaloriesConsumed': [data_to_update]  # Value you want to update
                }, index=[time_added])

                # Concatenate and sort the dataframe by the date index
                df = pd.concat([df, new_row])
                df = df.sort_index()

        try:
            df.to_csv(os.path.join(full_path, 'final_df.csv'), index=True, mode='w', na_rep='')
            sys.stdout.flush()
            print("File saved successfully")
        except Exception as e:
            print(f"Error saving final_df: {e}")


def save_tss_values_for_dashboard(file_path, tss, atl, ctl, tsb):
    full_path = get_full_path(file_path)
    tss.to_csv(os.path.join(full_path, 'tss.csv'), index=True)
    ctl.to_csv(os.path.join(full_path, 'ctl.csv'), index=True)
    atl.to_csv(os.path.join(full_path, 'atl.csv'), index=True)
    tsb.to_csv(os.path.join(full_path, 'tsb.csv'), index=True)


def load_tss_values_for_dashboard(file_path):
    full_path = get_full_path(file_path)
    tss = pd.read_csv(os.path.join(full_path, 'tss.csv'), index_col=0)
    ctl = pd.read_csv(os.path.join(full_path, 'ctl.csv'), index_col=0)
    atl = pd.read_csv(os.path.join(full_path, 'atl.csv'), index_col=0)
    tsb = pd.read_csv(os.path.join(full_path, 'tsb.csv'), index_col=0)

    return tss, atl, ctl, tsb


def clean_data_basic(dfs, date_cols):
    # Threshold of % of NaN's per column we want to accept
    threshold = 0.5

    for df_name, df in dfs.items():

        # REMOVED DROP_HIGH_NA_COLUMNS CUZ WEIRD BEHAVIOUR, WILL NEED TO UPDATE CUZ NO NEED

        # Remove columns with high > threshold % NaN values
        # drop_high_na_columns(df, threshold) # NOTE: - REMOVED THIS ONE

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
    df['TimeTotalInHours'] = pd.to_timedelta(df['TimeTotalInHours']).dt.total_seconds() / 3600  # Convert to Hours

    # Convert relevant columns to numeric (remove commas, etc.)
    df['DistanceInMeters'] = pd.to_numeric(df['DistanceInMeters'].str.replace(',', '.'), errors='coerce')
    df['Calories'] = pd.to_numeric(df['Calories'], errors='coerce')

    # Drop rows with NaN values in critical columns
    df = df.dropna(subset=['DistanceInMeters', 'Calories', 'TimeTotalInHours', 'HeartRateAverage'])

    # df = df[df['DistanceInMeters']>0].copy() # NOTE: not needed since, i will be using only TotalDuration or TimeTotalInHours

    return df


def print_performances(rmse_results):
    print()
    print()
    # Printing the performance metrics
    print("Performance Metrics:")

    for result in rmse_results:
        model_name = result['name']
        rmse_value = result['rmse']
        print(f"{model_name} RMSE: {rmse_value}")

    print()
    print()


def aggregate_by_date(cal_estimated_df, cal_calculated_df, activities):
    # HAD IT BEEN A NORMAL COLUMN - meaning 'Date' not as index, WE WOULD HAVE NEEDED TO DO THE FOLLOWING
    # activities['Date'] = pd.to_datetime(activities['Date']).dt.normalize()
    cal_estimated_df.index = pd.to_datetime(cal_estimated_df.index).normalize()
    cal_calculated_df.index = pd.to_datetime(cal_calculated_df.index).normalize()
    activities.index = pd.to_datetime(activities.index).normalize()

    ### FOR cal_estimated_df ###
    def weighted_heart_rate_average(df):
        # Choose the duration column: use 'TimeTotalInHours' if available, else 'PlannedDuration'
        duration_col = df['TimeTotalInHours'].fillna(df['PlannedDuration']).fillna(0)

        # Perform the weighted average
        total_duration = duration_col.sum()
        if total_duration > 0:
            weighted_hr = (df['HeartRateAverage'] * duration_col).sum() / total_duration # NOTE: MAYBE THIS IS NOT NEEDED ANYMORE
        else:
            weighted_hr = 0  # Handle the case where no valid durations are present
        return weighted_hr

    def concatenate_workout_types(series):
        return ', '.join(series.dropna().unique())

    def aggregate_group(group):
        # Special aggregation
        special_agg = pd.Series({
            'WorkoutType': concatenate_workout_types(group['WorkoutType']),
            'HeartRateAverage': weighted_heart_rate_average(group)
        })

        # Default aggregation (sum for other columns)
        default_agg = group.drop(columns=['WorkoutType', 'HeartRateAverage']).sum()

        # Combine special and default aggregations
        return pd.concat([special_agg, default_agg])


    # Apply aggregation
    cal_estimated_df_agg = cal_estimated_df.groupby(cal_estimated_df.index).apply(aggregate_group)




    ### FOR cal_calculated_df ###
    # Define the special aggregation for 'TotalPassiveCalories' (take the last value)
    special_aggregation_calculated = {
        'TotalPassiveCal': 'last'
    }

    # Default aggregation: sum for all other columns
    default_aggregation_calculated = {col: 'sum' for col in cal_calculated_df.columns if col != 'TotalPassiveCal'}

    # Merge the aggregation rules
    aggregation_rules_calculated = {**default_aggregation_calculated, **special_aggregation_calculated}

    # Apply aggregation
    cal_calculated_df_agg = cal_calculated_df.groupby('Date').agg(aggregation_rules_calculated)



    # cal_estimated_df_agg = cal_estimated_df.groupby('Date').agg('sum')
    # cal_calculated_df_agg = cal_calculated_df.groupby('Date').agg('sum')
    activities_agg = activities.groupby('Date').agg('sum')

    return cal_estimated_df_agg, cal_calculated_df_agg, activities_agg


def filter_workouts_and_remove_nans(df, given_date = GIVEN_DATE):
    columns_to_keep_workouts = ['WorkoutType', 'Title', 'WorkoutDescription', 'CoachComments', 'HeartRateAverage', 'TimeTotalInHours', 'DistanceInMeters', 'PlannedDuration', 'PlannedDistanceInMeters']
    df = df[columns_to_keep_workouts].copy()

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

        # Fill NaN values in object columns with an empty string
    object_cols = w_df.select_dtypes(include=['object']).columns
    w_df[object_cols] = w_df[object_cols].fillna('')

    return w_df


def process_data(user_data, workouts=None):
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
    # For Workouts and Activities For the moment
    clean_data_basic(dataframes, date_columns)

    ### WORKOUTS
    w_df = filter_workouts_and_remove_nans(dataframes['workouts'])

    # Calculate TSS per discipline and TOTAL TSS
    w_df = calculate_total_tss(w_df) # FIXME: i am creating this only once, despite updating the df with new workouts, as below, DON'T KNOW IF IT NEEDS UPDATING

    # # Calculate ATL, CTL, TSB from TSS
    tss_df, atl_df, ctl_df, tsb_df = calculate_metrics_from_tss(w_df) # FIXME: i am creating this only once, despite updating the df with new workouts

    # ACTIVITIES
    activities_df = clean_activities(dataframes['activities'])


    def micro_agression(work_df, acti_df):
        full_path = get_full_path('data/processed/csv/')
        work_df.to_csv(os.path.join(full_path, 'workouts_to_process_df.csv'), na_rep='')
        acti_df.to_csv(os.path.join(full_path, 'activities_to_process_df.csv'), na_rep='')
    # micro_agression(w_df, activities_df)

    # Separate past and future workouts
    past_workouts_df = w_df.loc[w_df.index < GIVEN_DATE]
    future_workouts_df = w_df.loc[w_df.index >= GIVEN_DATE]

    # workout_type = "with WorkoutType"
    workout_type = "duration with WorkoutType"
    # workout_type = "without WorkoutType"
    # Estimate Total Calories from Models

    if workout_type == "duration with WorkoutType":
        w_df_calories_estimated, rmse_results = estimate_calories_with_duration(activities_df, past_workouts_df, future_workouts_df)
    else:
        w_df_calories_estimated, rmse_results = estimate_calories(activities_df, past_workouts_df, future_workouts_df, workout_type)

    print_performances(rmse_results)

    # Calculate Total Calories from TSS
    w_df_calories_calculated = calculate_total_calories(user_data, df=w_df)

    aggregate_by_date_path = False
    final_columns = ['WorkoutType', 'Title', 'WorkoutDescription', 'CoachComments', 'HeartRateAverage', 'TimeTotalInHours',
                     'DistanceInMeters', 'PlannedDuration', 'PlannedDistanceInMeters', 'Run_Cal', 'Bike_Cal', 'Swim_Cal',
                     'TotalPassiveCal', 'CalculatedActiveCal', 'EstimatedActiveCal', 'Calories', 'CaloriesSpent', 'CaloriesConsumed']
    if aggregate_by_date_path:
        ### DATAFRAMES AGGREGATED BY DATE ###
        w_df_cal_est, w_df_cal_calc, activities_df = aggregate_by_date(w_df_calories_estimated, w_df_calories_calculated, activities_df)

        w_df_calories_estimated_plus_calculated_agg = pd.concat([w_df_cal_est, w_df_cal_calc], axis=1, join='inner')

        final_df = pd.concat([w_df_calories_estimated_plus_calculated_agg, activities_df['Calories']], axis=1)
        final_df = final_df.loc[:,~final_df.columns.duplicated()]

        final_df.index = pd.to_datetime(final_df.index)
        final_df.index = final_df.index.date

        final_df = final_df.reindex(columns=final_columns, fill_value=0.0)

        numeric_cols = final_df.select_dtypes(include=['float64', 'int64']).columns
        final_df[numeric_cols] = final_df[numeric_cols].fillna(0.0)
        final_df = final_df.drop(columns=['PlannedDuration', 'PlannedDistanceInMeters'])

        return tss_df, atl_df, ctl_df, tsb_df, w_df_calories_estimated_plus_calculated_agg, activities_df, final_df
    else:
        ### DATAFRAMES NOT AGGREGATED BY DATE ###
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


        return tss_df, atl_df, ctl_df, tsb_df, w_df_calories_estimated_plus_calculated, activities_df, final_df
