# load_and_update_final_csv_helper

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
from src.tss_calculations import * # NOTE: WHY IT WORKED WITH .tss_calculations before
from src.calorie_calculations import *
from src.calorie_estimation_models import *


def save_dataframe(df, full_path):
    try:
        df.to_csv(os.path.join(full_path, 'final_df.csv'), index=True, mode='w', na_rep='')
        print("File saved successfully")
    except Exception as e:
        print(f"Error saving final_df: {e}")

def update_or_add_row(df, date_str, workout_type, updates):
    """
    Update the row with the given `date_str` and `workout_type`.
    If the row does not exist, a new one will be created with default values,
    and the `updates` will be applied.

    Parameters:
    - df: The DataFrame to update.
    - date_str: The index (date) to check or add.
    - workout_type: The workout type to check.
    - updates: A dictionary of columns to update with their corresponding values.

    Returns:
    The updated DataFrame.
    """
    existing_row_mask = (df.index == date_str) & (df['WorkoutType'] == workout_type)

    # If row exists, update it
    if not df[existing_row_mask].empty:
        df.loc[existing_row_mask, updates.keys()] = list(updates.values())
    else:
        # Create a new row with default values using create_default_row
        new_row = create_default_row(date_str)

        # Apply the workout type and other updates to the new row
        new_row['WorkoutType'] = workout_type
        for col, value in updates.items():
            new_row[col] = value

        # Concatenate the new row and sort the DataFrame
        df = pd.concat([df, new_row])
        df = df.sort_index()

    return df


def process_data_to_update(df, data_to_update):
    for i, row in data_to_update.iterrows():
        date_str = row['Date']
        updates = {
            'Title': row['Title'],
            'WorkoutDescription': row['WorkoutDescription'],
            'CoachComments': row['CoachComments'],
            'TimeTotalInHours': row['duration'],
            'ComplianceStatus': row['compliance_status'], # training peaks
            'TSS': row['tss'] # training peaks
        }
        df = update_or_add_row(df, date_str, row['WorkoutType'], updates)
    return df


def process_activity_dict(df, activity_dict):
    for activity in activity_dict:
        date_str = pd.to_datetime(activity['Date']).strftime('%Y-%m-%d')
        updates = {
            'Title': activity.get('Title', ''),
            'WorkoutDescription': activity.get('WorkoutDescription', ''),
            'CoachComments': activity.get('CoachComments', ''),
            'TimeTotalInHours': activity.get('duration', 0) / 60,
            'CaloriesSpent': activity.get('CaloriesSpent', 0.0),
            'ComplianceStatus': activity.get('compliance_status', ''), # training peaks
            'TSS': activity.get('tss', 0.0) # training peaks
        }
        df = update_or_add_row(df, date_str, activity.get('WorkoutType', ''), updates)
    return df


def create_default_row(time_added):
    """
    Create a new DataFrame row with default values (0.0 for floats, '' for objects).

    Parameters:
    - time_added: index value (usually a timestamp) for the new row.

    Returns:
    A DataFrame row with default values and Date index
    """
    default_values = {
        'WorkoutType': '',
        'Title': '',
        'WorkoutDescription': '',
        'CoachComments': '',
        'ComplianceStatus': '',
        'HeartRateAverage': 0.0,
        'TimeTotalInHours': 0.0,
        'DistanceInMeters': 0.0,
        'Run_Cal': 0.0,
        'Bike_Cal': 0.0,
        'Swim_Cal': 0.0,
        'TotalPassiveCal': 0.0,
        'CalculatedActiveCal': 0.0,
        'EstimatedActiveCal': 0.0,
        'Calories': 0.0,
        'CaloriesSpent': 0.0,
        'CaloriesConsumed': 0.0,
        # 'rTSS Calculated': 0.0,
        # 'cTSS Calculated': 0.0,
        # 'sTSS Calculated': 0.0,
        # 'TOTAL TSS': 0.0
        'TSS': 0.0
    }
    return pd.DataFrame(default_values, index=[time_added])
