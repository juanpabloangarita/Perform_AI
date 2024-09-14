import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add the 'src' directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) # DIDN'T WORK, WHY? WARNING

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located - pages
dir_script_dir = os.path.dirname(script_dir) #directory = streamlit_app
dir_script_dir = os.path.dirname(dir_script_dir) #src
sys.path.append(dir_script_dir)

from src.data_processing import *
from src.calorie_calculations import *

# Step 1: Choose the goal (Weight Loss or Maintenance)
st.title("Calorie Calculation Dashboard")

goal = st.radio(
    "What is your goal?",
    ("Lose weight", "Maintain weight")
)

if goal == "Lose weight":
    st.write("To lose weight, aim to have a calorie deficit.")
elif goal == "Maintain weight":
    st.write("To maintain weight, balance your calorie intake and expenditure.")

# Step 2: Select exercise type, duration, and average heart rate in a single row
st.write("## Exercise Information")

# Create a layout with three columns
col1, col2, col3 = st.columns(3)

with col1:
    exercise_type = st.selectbox("Type of exercise:", ["Running", "Cycling", "Swimming"])

with col2:
    duration = st.number_input(
        "Duration (hours)",
        min_value=0.1, max_value=12.0, value=1.0
    )

with col3:
    avg_heart_rate = st.number_input(
        "Average Heart Rate (bpm)",
        min_value=30, max_value=220, value=120
    )

# Step 3: Choose to use saved user data or input new data
st.write("## User Information")

user_option = st.radio(
    "Select user type:",
    ("Use saved data", "New user")
)

# Organize user data input fields into a more compact layout
if user_option == "New user":
    st.write("Please provide your information:")

    col1, col2 = st.columns(2)

    with col1:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=st.session_state['user_data']['weight'])
        height = st.number_input("Height (cm)", min_value=120, max_value=250, value=st.session_state['user_data']['height'])
        age = st.number_input("Age (years)", min_value=10, max_value=100, value=st.session_state['user_data']['age'])

    with col2:
        gender = st.selectbox("Gender", options=["male", "female"], index=0 if st.session_state['user_data']['gender'] == 'male' else 1)
        vo2_max = st.number_input("VO2 Max", min_value=20, max_value=90, value=st.session_state['user_data']['vo2_max'])
        resting_hr = st.number_input("Resting Heart Rate", min_value=30, max_value=120, value=st.session_state['user_data']['resting_hr'])

    # Update session state with new values
    st.session_state['user_data'] = {
        'weight': weight,
        'height': height,
        'age': age,
        'gender': gender,
        'vo2_max': vo2_max,
        'resting_hr': resting_hr
    }

# Use saved user data if selected
if user_option == "Use saved data":
    weight = st.session_state['user_data']['weight']
    height = st.session_state['user_data']['height']
    age = st.session_state['user_data']['age']
    gender = st.session_state['user_data']['gender']
    vo2_max = st.session_state['user_data']['vo2_max']
    resting_hr = st.session_state['user_data']['resting_hr']

# Step 4: Process Data and Calculate Calories
st.write("## Calculate Calories Burned")

# Calculate calories burned
calories_burned = calculate_total_calories(
    weight=weight, height=height, age=age, gender=gender,
    vo2_max=vo2_max, resting_hr=resting_hr)

# Simulate a DataFrame to represent the workout data for this session
w_df = pd.DataFrame({
    "exercise_type": [exercise_type],
    "duration_hours": [duration],
    "avg_heart_rate": [avg_heart_rate],
    "TSS": [np.nan],  # You can calculate TSS based on the inputs if needed
    "Calories": [np.nan]
})

st.write(f"Total calories burned: {calories_burned:.2f} kcal")

# Suggest calorie intake based on goal
if goal == "Lose weight":
    st.write("To achieve a calorie deficit, you should consume fewer calories than you burn.")
elif goal == "Maintain weight":
    st.write("To maintain your weight, consume a similar number of calories as you burn.")
