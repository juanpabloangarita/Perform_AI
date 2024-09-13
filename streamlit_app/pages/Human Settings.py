import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
# Add the 'src' directory to sys.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) # DIDN'T WORK, WHY? WARNING
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located - pages
dir_script_dir = os.path.dirname(script_dir) #directory = streamlit_app
dir_script_dir = os.path.dirname(dir_script_dir) #src
sys.path.append(dir_script_dir)

from src.data_processing import *
from src.calorie_calculations import *
from src.user_data import *
from src.user_data_cloud import *
from params import *

# Display current user data in editable form
st.title("Human Settings")

# Step 1: Show current goal (Weight Loss or Maintenance)
st.write("## Current Goal")
goal = st.radio(
    "What is your goal?",
    ("Lose weight", "Maintain weight"),
    index=0 if st.session_state['user_data']['goal'] == 'Lose weight' else 1
)

# Update the goal in session state when modified
st.session_state['user_data']['goal'] = goal

# Step 2: Show and modify user-specific information
st.write("## Your Information")

# Create a layout with two columns for user information
col1, col2 = st.columns(2)

with col1:
    weight = st.number_input(
        "Weight (kg)",
        min_value=30,
        max_value=200,
        value=int(st.session_state['user_data'].get('weight', 80)
                  if pd.notna(st.session_state['user_data'].get('weight'))
                  else 80)  # Handle NaN or None
    )
    height = st.number_input(
        "Height (cm)",
        min_value=120,
        max_value=250,
        value=int(st.session_state['user_data'].get('height', 183)
                  if pd.notna(st.session_state['user_data'].get('height'))
                  else 183)  # Handle NaN or None
    )
    age = st.number_input(
        "Age (years)",
        min_value=10,
        max_value=100,
        value=int(st.session_state['user_data'].get('age', 41)
                  if pd.notna(st.session_state['user_data'].get('age'))
                  else 41)  # Handle NaN or None
    )

with col2:
    gender = st.selectbox(
        "Gender",
        options=["male", "female"],
        index=0 if st.session_state['user_data']['gender'] == 'male' else 1
    )
    vo2_max = st.number_input(
        "VO2 Max",
        min_value=20,
        max_value=90,
        value=int(st.session_state['user_data'].get('vo2_max', 50)
                    if pd.notna(st.session_state['user_data'].get('vo2_max'))
                    else 50)  # Handle NaN or None
    )
    resting_hr = st.number_input(
        "Resting Heart Rate",
        min_value=30,
        max_value=120,
        value=int(st.session_state['user_data'].get('resting_hr', 42)
                  if pd.notna(st.session_state['user_data'].get('resting_hr'))
                  else 42)  # Handle NaN or None
    )


if st.button('Update'):
    passive_calories, bmr = calculate_total_calories(weight, height, age, gender, vo2_max, resting_hr, 'streamlit')
    # Update the session state when user modifies the information
    st.session_state['user_data'] = {
        'weight': weight,
        'height': height,
        'age': age,
        'gender': gender,
        'vo2_max': vo2_max,
        'resting_hr': resting_hr,
        'goal': goal,
        'BMR': bmr,
        'passive_calories': passive_calories
    }

    if CLOUD_ON == 'yes':
        # Call the update_user_data_cloud function by merging both dictionaries
        update_user_data_cloud(**{**st.session_state['user_data'], 'username': st.session_state['username']})
    else:
        update_user_data(**{**st.session_state['user_data'], 'username': st.session_state['username']})
    # Display a message indicating successful update
    st.success("Your information has been updated!")
