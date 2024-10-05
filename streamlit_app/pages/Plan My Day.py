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

            submit_button = st.form_submit_button(label='Add Activity')
            if submit_button:
                temp_activity_dict = {
                    activity: {
                        'duration': duration,
                        'calories_spent': calories_spent,
                        'heart_rate': heart_rate,
                        'distance': distance
                    }
                }

                st.success(f"Added {duration} minutes of {activity} with {calories_spent} calories, {heart_rate} bpm, and {distance} meters")
                load_and_update_final_csv('data/processed/csv/', "input_activities", GIVEN_DATE, temp_activity_dict)
        # Ensure that the form has enough space, add a placeholder if necessary
        with st.empty():
            pass


    with col2:
        st.write("#### Today's Trainings")

        # Add a container-like background for consistency
        with st.container():
            df = load_and_update_final_csv('data/processed/csv/', "plan_my_day")
            total_active_calories = 0

            if GIVEN_DATE in df.index:
                data_for_date = df.loc[GIVEN_DATE]

                if isinstance(data_for_date, pd.Series):
                    # Handle case when there's only one entry (Series)
                    data_for_date = data_for_date.to_frame().T

                for i, row in data_for_date.iterrows():
                    activity = row['WorkoutType']
                    avg_heart_rate = row['HeartRateAverage']
                    total_duration = row['TimeTotalInHours']
                    total_distance = row['DistanceInMeters']
                    calories_manually_input = row['CaloriesSpent']
                    calories_estimated = row['EstimatedActiveCal']
                    total_calories = calories_manually_input if calories_manually_input > 0 else calories_estimated
                    total_active_calories += total_calories
                    total_duration_hours = int(total_duration)  # Get the whole number part as hours
                    total_duration_minutes = int((total_duration - total_duration_hours) * 60)  # Get the fractional part as minutes
                    converted_total_duration = f"{total_duration_hours}h {total_duration_minutes}min"

                    # Using bullet points with icons for activities
                    st.write(f"🟢 **{activity}**")
                    st.write(f"  - {total_distance:.1f} meters in {converted_total_duration} minutes with an Average HR of {avg_heart_rate:.1f} bpm")
                    st.write(f"  - **Total Calories:** {total_calories:.1f} kcal")
            else:
                st.write("No activities added today.")

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
                st.success(f"Added {calories} calories")
                load_and_update_final_csv('data/processed/csv/', 'calories_consumed', GIVEN_DATE, calories)

        # Ensure space is balanced between columns
        with st.empty():
            pass


    # Second column: Display calories and remaining calories
    with col2:
        st.write("#### Today's Calories")
        df = load_and_update_final_csv('data/processed/csv/', "plan_my_day")

        calories_data = df.loc[GIVEN_DATE, 'CaloriesConsumed']

        if isinstance(calories_data, pd.Series):
            # If it's a Series, drop NaNs and take the first non-empty value
            calories_consumed = calories_data.dropna().iloc[0] if not calories_data.dropna().empty else None
        else:
            # If it's a single value (numpy.float64), use it directly (it could be NaN)
            calories_consumed = calories_data if not pd.isna(calories_data) else None

        # Display calories consumed
        st.write(f"**Calories Consumed Today:** {calories_consumed} kcal")

        # Calculate total daily calorie needs
        total_daily_calories = st.session_state['user_data']['passive_calories'] + total_active_calories # NOTE: NOT SURE ABOUT total_active_calories that was created before
        calories_remaining = total_daily_calories - calories_consumed
        st.write(f"**Calories To Consume Today:** {calories_remaining} kcal")

        # FIXME: Calculations wrong, above and below
        st.progress(min(calories_consumed / total_daily_calories, 1.0))  # To ensure progress stays between 0 and 1

        # Provide guidance on managing remaining calories
        if calories_remaining > 0:
            st.write("🟢 You have room for more food today!")
        else:
            st.write("🔴 You've reached or exceeded your calorie limit for today. Consider balancing your intake.")

        # Ensure space is balanced with the first column
        with st.empty():
            pass
