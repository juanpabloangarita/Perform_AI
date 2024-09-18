import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

from datetime import datetime

from PIL import Image
# Add the 'src' directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) # DIDN'T WORK, WHY? WARNING

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located - pages
dir_script_dir = os.path.dirname(script_dir) #directory = streamlit_app
dir_script_dir = os.path.dirname(dir_script_dir) #src
sys.path.append(dir_script_dir)

from src.data_processing import load_and_update_final_csv
from src.calorie_calculations import *


# Define a function to calculate active calories based on activities
def calculate_active_calories(activities):
    total_calories = 0
    for activity in activities.keys():
        total_calories += activities[activity]['calories_spent']

    return total_calories


# Initialize or load session state
if 'activities' not in st.session_state:
    st.session_state['activities'] = {}
if 'calories_consumed' not in st.session_state:
    st.session_state['calories_consumed'] = 0

st.title("Plan My Day")

with st.container(border=True):
    # Create two columns
    col1, col2 = st.columns(2)


    # First column: Improved Input Form for Activities
    with col1:
        st.write("### Input Today's Training")
        activity_options = ['Run', 'Bike', 'Swim']  # TODO: Calculate strength training calories

        with st.form(key='activity_form'):
            # Create a grid layout for input fields
            col11, col12 = st.columns(2)

            with col11:
                activity = st.selectbox("Select Activity", activity_options)

            with col12:
                duration = st.number_input("Duration (minutes)", min_value=0, step=1)

            with col11:
                heart_rate = st.number_input("Heart Rate (bpm)", min_value=0, step=1)

            with col12:
                distance = st.number_input("Distance (meters)", min_value=0.0, step=0.1)

            with col11:
                calories_spent = st.number_input("Calories Spent", min_value=0.0, step=0.1)

            # Add empty space before the button to push it down
            # st.write("")  # Creates an empty line
            # st.write("")  # Creates additional empty space
            submit_button = st.form_submit_button(label='Add Activity')
            if submit_button:
                temp_activity_dict = {}

                # Store the new activity details in the temporary dictionary
                temp_activity_dict[activity] = {
                    'duration': duration,
                    'calories_spent': calories_spent,
                    'heart_rate': heart_rate,
                    'distance': distance
                }

                if activity in st.session_state['activities']:
                    # /* TODO: IN THE FOLLOWING CODE, I AM ONLY AVERAGING HEART RATE, IF for example 'Bike' already exists, and so, averaging only with 'Bike' activity
                    # Retrieve previous data
                    prev_data = st.session_state['activities'][activity]
                    prev_duration = prev_data['duration']
                    prev_heart_rate = prev_data['heart_rate']

                    # Compute new average heart rate
                    new_avg_heart_rate = (
                        (prev_heart_rate * prev_duration + heart_rate * duration) /
                        (prev_duration + duration)
                    )
                    # */

                    # Update the session state with new values
                    st.session_state['activities'][activity] = {
                        'duration': prev_duration + duration,
                        'calories_spent': prev_data['calories_spent'] + calories_spent,
                        'heart_rate': new_avg_heart_rate,
                        'distance': prev_data['distance'] + distance
                    }
                else:
                    st.session_state['activities'][activity] = {
                        'duration': duration,
                        'calories_spent': calories_spent,
                        'heart_rate': heart_rate,
                        'distance': distance
                    }
                st.success(f"Added {duration} minutes of {activity} with {calories_spent} calories, {heart_rate} bpm, and {distance} meters")
                timestamp = datetime.now().strftime('%Y-%m-%d')
                load_and_update_final_csv('data/processed/csv/', "input_activities", timestamp, temp_activity_dict)
        # Ensure that the form has enough space, add a placeholder if necessary
        with st.empty():
            pass


   # Second column: Display activities and active calories with enhanced UI
    with col2:
        st.write("#### Today's Trainings")

        # Add a container-like background for consistency
        with st.container():
            if st.session_state['activities']:
                for activity, details in st.session_state['activities'].items():
                    # Extract details
                    total_duration = details['duration']
                    total_calories = details['calories_spent']
                    avg_heart_rate = details['heart_rate']
                    total_distance = details['distance']

                    # Using bullet points with icons for activities
                    st.write(f"🟢 **{activity}**")
                    st.write(f"  - {total_distance:.1f} meters in {total_duration} minutes with an Average HR of {avg_heart_rate:.1f} bpm")
                    st.write(f"  - **Total Calories:** {total_calories:.1f} kcal")
            else:
                st.write("No activities added today.")

            # Calculate and display total active calories
            total_active_calories = calculate_active_calories(st.session_state['activities'])
            st.write(f"**Total Active Calories Burned Today:** {total_active_calories:.1f} kcal")

            # Ensure that the form has enough space, add a placeholder if necessary
            with st.empty():
                pass


with st.container(border=True):
    # Create two columns for layout
    col1, col2 = st.columns(2)

    # First column: Input calories consumed
    with col1:
        st.write("### Nutrition")

        # Form to input calories consumed
        with st.form(key='food_form'):
            calories = st.number_input("Calories Consumed", min_value=0, step=1)
            submit_button = st.form_submit_button(label='Add Calories')

            if submit_button:
                st.session_state['calories_consumed'] += calories
                st.success(f"Added {calories} calories") # NOTE: MAYBE I ALWAYS HAVE TO PRINT STUFF ON THE SCREEN TO HANDLE STREAMLIT STATES
                timestamp = datetime.now().strftime('%Y-%m-%d')
                load_and_update_final_csv('data/processed/csv/', 'calories_consumed', timestamp, calories)

        # Ensure space is balanced between columns
        with st.empty():
            pass


    # Second column: Display calories and remaining calories
    with col2:
        st.write("#### Today's Calories")

        # Display calories consumed
        st.write(f"**Calories Consumed Today:** {st.session_state['calories_consumed']} kcal")

        # Calculate total daily calorie needs
        total_daily_calories = st.session_state['user_data']['passive_calories'] + total_active_calories
        calories_remaining = total_daily_calories - st.session_state['calories_consumed']
        st.write(f"**Calories Remaining for Today:** {calories_remaining} kcal")

        # FIXME: Calculations wrong, above and below
        st.progress(min(st.session_state['calories_consumed'] / total_daily_calories, 1.0))  # To ensure progress stays between 0 and 1

        # Provide guidance on managing remaining calories
        if calories_remaining > 0:
            st.write("🟢 You have room for more food today!")
        else:
            st.write("🔴 You've reached or exceeded your calorie limit for today. Consider balancing your intake.")

        # Ensure space is balanced with the first column
        with st.empty():
            pass
