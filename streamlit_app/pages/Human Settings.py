import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add the 'src' directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
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
st.write("### Goal Settings")
goal = st.radio(
    "Select your current goal:",
    ("Lose weight", "Maintain weight"),
    index=0 if st.session_state['user_data']['goal'] == 'Lose weight' else 1,
    help="Your goal affects how we calculate your daily calorie needs."
)

# Update the goal in session state when modified
st.session_state['user_data']['goal'] = goal

with st.container(border=True):
    # Step 2: Daily Metrics Section
    st.write("### Daily Metrics")
    st.write("These metrics can change frequently based on your daily activity and body conditions.")

    # Create a layout with two columns for daily metrics
    daily_col1, daily_col2 = st.columns(2)

    with daily_col1:
        st.session_state['user_data']['weight'] = st.number_input(
            "Weight (kg)",
            min_value=30,
            max_value=200,
            value=int(st.session_state['user_data']['weight']),
            help="Your current weight. Helps in calculating calorie needs."
        )
        st.session_state['user_data']['age'] = st.number_input(
            "Age (years)",
            min_value=10,
            max_value=100,
            value=int(st.session_state['user_data']['age']),
            help="Your age affects your basal metabolic rate (BMR)."
        )

    with daily_col2:
        st.session_state['user_data']['resting_hr'] = st.number_input(
            "Resting Heart Rate",
            min_value=30,
            max_value=120,
            value=int(st.session_state['user_data']['resting_hr']),
            help="Your resting heart rate affects your overall fitness level."
        )

with st.container(border=True):
    # Step 3: Calories Section (Passive Calories and BMR)
    st.write("### Calories Information")
    st.write("Passive calories and BMR are key metrics to track your energy expenditure.")

    cal_col1, cal_col2 = st.columns(2)

    with cal_col1: # FIXME: is not user friendly to show false values from the beginning, people will think these are their values when it is not.
        st.session_state['user_data']['passive_calories'] = st.number_input(
            "Passive Calories",
            value=float(st.session_state['user_data']['passive_calories']),
            help="Your estimated passive calories for the day, including basic daily activities."
        )

    with cal_col2: # FIXME: is not user friendly to show false values from the beginning, people will think these are their values when it is not.
        st.session_state['user_data']['BMR'] = st.number_input(
            "BMR (Basal Metabolic Rate)",
            value=float(st.session_state['user_data']['BMR']),
            help="Your BMR is the amount of energy expended while at rest. It impacts your overall calorie needs."
        )

with st.container(border=True):
    # Step 4: Biometric Data Section
    st.write("### Biometric Data")
    st.write("These details are more stable and don’t change often.")

    bio_col1, bio_col2 = st.columns(2)

    with bio_col1:
        st.session_state['user_data']['height'] = st.number_input(
            "Height (cm)",
            min_value=50,
            max_value=250,
            value=int(st.session_state['user_data']['height']),
            help="Your height helps calculate your BMI and calorie requirements."
        )

    with bio_col2:
        st.session_state['user_data']['vo2_max'] = st.number_input(
            "VO2 Max",
            min_value=20,
            max_value=90,
            value=int(st.session_state['user_data']['vo2_max']),
            help="VO2 Max is a key indicator of cardiovascular fitness."
        )

    # Gender Input (Moved to dropdown for better UX)
    st.session_state['user_data']['gender'] = st.selectbox(
        "Gender",
        ["Male", "Female"],
        index=["Male", "Female"].index(st.session_state['user_data']['gender']),
        help="Select your gender for accurate calorie and fitness calculations."
    )

if st.button('Update Information'):
    passive_calories, bmr = calculate_total_calories(st.session_state['user_data'], 'streamlit')
    # Update the session state when user modifies the information
    st.session_state['user_data']['BMR'] = bmr
    st.session_state['user_data']['passive_calories'] = passive_calories
    # Update the session_state when the user changes their goal
    st.session_state['user_data']['goal'] = goal

    if CLOUD_ON == 'yes':
        # Call the update_user_data_cloud function by merging both dictionaries
        update_user_data_cloud(**{**st.session_state['user_data'], 'username': st.session_state['username']})
    else:
        update_user_data(**{**st.session_state['user_data'], 'username': st.session_state['username']})

    # Display a success message
    st.success("Your information has been successfully updated!")
