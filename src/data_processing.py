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

from src.calorie_calculations import *
from src.calorie_estimation_models import *
from src.calorie_estimation_models import estimate_calories_with_duration, load_model, estimate_calories_with_nixtla
from src.calorie_estimation_models_previous import estimate_calories, estimate_calories_with_duration_previous
from src.load_and_update_final_csv_helper import process_data_to_update, process_activity_dict, update_or_add_row, create_default_row
from src.data_loader.files_saving import FileSaver
from src.data_loader.get_full_path import get_full_path # TODO: ASK DINIS: -> for the file files_saving.py in data_loader, i did the equivalente, meaning from src. and it didn't work, meaning i did from data_loader.
from src.tss_calculations import * # TODO: WHY IT WORKED WITH .tss_calculations before ASK DINIS, RELATED TO  THE ONE BELOW
from src.data_loader.files_extracting import FileLoader
from params import CLOUD_ON


def load_and_update_final_csv(file_path, from_where, time_added=None, data_to_update=None):
    full_path = get_full_path(file_path)
    df = pd.read_csv(os.path.join(full_path, 'final_df.csv'), index_col=0, na_filter=False)

    if from_where in ['home', 'plan_my_day']:
        return df

    elif from_where == 'training_peaks' and isinstance(data_to_update, pd.DataFrame):
        df = process_data_to_update(df, data_to_update)

    elif from_where == 'training_peaks' and isinstance(data_to_update, list):
        df = process_activity_dict(df, data_to_update)

    elif from_where == 'plan_my_week':
        updates = {
            'TimeTotalInHours': pd.to_numeric(data_to_update.get('TimeTotalInHours', 0), errors='coerce')/60,
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
                'EstimatedActiveCal': details.get('estimated_calories', 0)  # Consistent key and default value
            }
            df = update_or_add_row(df, time_added, activity, updates)

    elif from_where == 'calories_consumed':
        if time_added in df.index:
            df.loc[time_added, 'CaloriesConsumed'] += data_to_update
        else: # NOTE: THIS HAPPENS ONLY WHEN THERE IS NO VALUE FOR THAT DAY AND I WANT TO ADD CONSUMED CALORIES not associated to any workout type cuz there is none
            new_row = create_default_row(time_added)
            new_row['CaloriesConsumed'] = data_to_update

            df = pd.concat([df, new_row]).sort_index()


    FileSaver().save_csv_files(df=df) # TODO: SAVE ALL WITH THE FileSaver() BELOW?
    # Calculate TSS per discipline and TOTAL TSS
    w_df = calculate_total_tss(df, 'load_and_update_final_csv')

    # # Calculate ATL, CTL, TSB from TSS
    w_df.index = pd.to_datetime(w_df.index)
    tss_df, atl_df, ctl_df, tsb_df = calculate_metrics_from_tss(w_df)
    FileSaver().save_tss_values_for_dashboard(tss_df, atl_df, ctl_df, tsb_df)


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
    before_df_cleaned = before_df[~(before_df['HeartRateAverage'].isna() & before_df['TimeTotalInHours'].isna())].copy() # NOTE: HERE IS THE PART THAT CAUSES THE WEIRD BEHAVIOUR. Explanation below
    # TODO: this because, the workouts that have been done, they all have a heart rate average and a time total in hours. However, they all have heart rate, when i export the csv. But when the date changes for tomorrow, the training that i had today, that i did, there is no info about it, and so, it is sort of removed. i have to
    # TODO: I have to update it with the training peaks thing. i need to find a way to have it on all the time. the real costs.
    # TODO: (BTW, I DON'T NEED TO REMOVE THE HEARTRATEAVERAGE.ISNA, since what's important for me is timetotalinhours only)

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
    sourcer = FileLoader()

    if CLOUD_ON == 'no':
        sourcer.load_initial_csv_files()

    sourcer.load_raw_and_final_dataframes('data/raw/csv')
    activities_df = sourcer.activities_raw



    if workouts is not None:
        workouts_df = workouts
    else:
        workouts_df = sourcer.workouts_raw



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
    w_df = calculate_total_tss(w_df, 'data_processing')

    # # Calculate ATL, CTL, TSB from TSS
    tss_df, atl_df, ctl_df, tsb_df = calculate_metrics_from_tss(w_df)

    # ACTIVITIES
    activities_df = clean_activities(dataframes['activities'])

    # workout_type = "with WorkoutType"
    workout_type = "duration with WorkoutType"
    # workout_type = "without WorkoutType"
    # Estimate Total Calories from Models

    # Separate past and future workouts
    past_workouts_df = w_df.loc[w_df.index < GIVEN_DATE]
    future_workouts_df = w_df.loc[w_df.index >= GIVEN_DATE]


    # if workout_type == "duration with WorkoutType":
    #     w_df_calories_estimated, rmse_results = estimate_calories_with_duration_previous(activities_df, past_workouts_df, future_workouts_df)
    # else:
    #     # FIXME: the following line comes from here from src.calorie_estimation_models_previous import estimate_calories
    #     w_df_calories_estimated, rmse_results = estimate_calories(activities_df, past_workouts_df, future_workouts_df, workout_type)


    X_activities = activities_df.rename(columns={'TimeTotalInHours': 'TotalDuration'}).copy()
    y_activities = activities_df['Calories']
    # Create and save models
    rmse_results = estimate_calories_with_duration(X_activities, y_activities)
    # Load preproc
    preprocessing_pipeline = load_model('preprocessing_pipeline')
    # Load linear model

    linear_model = load_model(BEST_MODEL)


    total_workouts = pd.concat([past_workouts_df, future_workouts_df])


    total_workouts[['TimeTotalInHours', 'DistanceInMeters']] = total_workouts[['TimeTotalInHours', 'DistanceInMeters']].fillna({
        'TimeTotalInHours': total_workouts['PlannedDuration'],
        'DistanceInMeters': total_workouts['PlannedDistanceInMeters']
        })
    # total_workouts = total_workouts.rename(columns={'TimeTotalInHours': 'TotalDuration'})
    total_workouts['TotalDuration'] = total_workouts['TimeTotalInHours']
    w_df_calories_estimated = total_workouts.copy()

    mask_total = total_workouts[['TotalDuration']].notna().all(axis=1)


    total_workouts_transformed = preprocessing_pipeline.transform(total_workouts[mask_total])


    w_df_calories_estimated.loc[mask_total, 'EstimatedActiveCal'] = linear_model.predict(total_workouts_transformed)

    FileSaver().save_during_process(w_df_calories_estimated, activities_df)

    ### NIXTLA ###
    future_workouts_for_nixtla = future_workouts_df.rename(columns={'PlannedDuration': 'TotalDuration'}).copy()

    forecast, sf = estimate_calories_with_nixtla(X_activities, y_activities, future_workouts_for_nixtla)


    # Step 1: Reset the index of w_df_calories_estimated and rename it to 'Date'
    w_df_calories_estimated = w_df_calories_estimated.reset_index().rename(columns={'index': 'Date'})

    # Step 2: Rename the 'ds' column in forecast to 'Date'
    forecast = forecast.rename(columns={'ds': 'Date'})

    # Step 3: Convert both 'Date' columns to datetime format if necessary
    w_df_calories_estimated['Date'] = pd.to_datetime(w_df_calories_estimated['Date'], format='%Y-%m-%d')
    forecast['Date'] = pd.to_datetime(forecast['Date'], format='%Y-%m-%d')

    # Step 4: Perform the merge on the 'Date' column
    w_df_calories_estimated = pd.merge(w_df_calories_estimated, forecast, on='Date', how='left')

    # Step 5: Set the 'Date' column as the index again
    w_df_calories_estimated.set_index('Date', inplace=True)

    # Now, w_df_calories_estimated contains the merged data with 'Date' as the index





    print_performances(rmse_results)
    # Calculate Total Calories from TSS
    w_df_calories_calculated = calculate_total_calories(user_data, df=w_df)

    aggregate_by_date_path = False
    final_columns = ['WorkoutType', 'Title', 'WorkoutDescription', 'CoachComments', 'HeartRateAverage', 'TimeTotalInHours',
                     'DistanceInMeters', 'PlannedDuration', 'PlannedDistanceInMeters', 'Run_Cal', 'Bike_Cal', 'Swim_Cal',
                     'TotalPassiveCal', 'CalculatedActiveCal', 'EstimatedActiveCal', 'AutoARIMA',  'AutoARIMA-lo-95',  'AutoARIMA-hi-95', 'Calories', 'CaloriesSpent', 'CaloriesConsumed']
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
        final_df['ComplianceStatus'] = ''
        final_df['TSS'] = 0.0

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
        final_df['ComplianceStatus'] = ''
        final_df['TSS'] = 0.0 # NOTE: probably this could be inserted as final columns, and used the reindex fill_value = 0.0 or a dictionary for this and before line


        return tss_df, atl_df, ctl_df, tsb_df, w_df_calories_estimated_plus_calculated, activities_df, final_df
