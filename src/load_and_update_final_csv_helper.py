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
    existing_row_mask = (df.index == date_str) & (df['WorkoutType'] == workout_type)
    if not df[existing_row_mask].empty:
        df.loc[existing_row_mask, updates.keys()] = list(updates.values())
    else:
        new_row = pd.DataFrame({**updates, 'WorkoutType': workout_type}, index=[date_str])
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
            'TimeTotalInHours': row['duration']
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
            'CaloriesSpent': activity.get('CaloriesSpent', 0.0)
        }
        df = update_or_add_row(df, date_str, activity.get('WorkoutType', ''), updates)
    return df
