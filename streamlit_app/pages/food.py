import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

from datetime import datetime
from PIL import Image

# Add the 'src' directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located - pages
dir_script_dir = os.path.dirname(script_dir)  # directory = streamlit_app
dir_script_dir = os.path.dirname(dir_script_dir)  # src
sys.path.append(dir_script_dir)

from src.data_processing import load_and_update_final_csv
from src.calorie_estimation_models import load_model
from params import BEST_MODEL, GIVEN_DATE

# Initialize session state for meals if not already done
if 'meals' not in st.session_state:
    st.session_state['meals'] = {
        'Breakfast': [],
        'Lunch': [],
        'Dinner': [],
        'Snack': []  # 'Gouté' can be considered as 'Snack'
    }

# Load foods_df
@st.cache_data
def load_foods_df():
    # Replace 'foods_df.csv' with the actual path to your foods_df CSV file
    return load_and_update_final_csv('data/processed/csv/', 'foods_df')

foods_df = load_foods_df()

st.title("Plan My Day")

# Sidebar for user settings (optional)
# You can add user settings like daily calorie goals here

with st.container():
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
                if duration > 0:
                    try:
                        # Load the preprocessing pipeline
                        preprocessing_pipeline = load_model('preprocessing_pipeline')
                        if preprocessing_pipeline is None:
                            st.error("Preprocessing pipeline not found. Please train the model first.")
                            st.stop()

                        # Create a DataFrame with required features
                        input_data = pd.DataFrame({
                            'WorkoutType': [activity],
                            'TotalDuration': [duration / 60]  # Convert minutes to hours
                        })

                        # Transform the input data
                        duration_transformed = preprocessing_pipeline.transform(input_data)

                        # Load the trained linear regression model
                        linear_model = load_model(BEST_MODEL)  # FIXME: UPLOAD THE CORRECT MODEL EACH TIME AUTOMATICALLY
                        if linear_model is None:
                            st.error("Linear Regression model not found. Please train the model first.")
                            st.stop()

                        # Predict the estimated calories
                        estimated_calories = linear_model.predict(duration_transformed)[0]
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
                        estimated_calories = 0
                else:
                    st.warning("Duration must be greater than 0 to estimate calories.")
                    estimated_calories = 0  # Or set to None if preferred

                # Create the activity dictionary with consistent key naming
                temp_activity_dict = {
                    activity: {
                        'duration': duration,
                        'calories_spent': calories_spent,
                        'heart_rate': heart_rate,
                        'distance': distance,
                        'estimated_calories': estimated_calories  # Consistent key
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







# Load the foods dataframe
foods_df = load_foods_df()

with st.container():
    # Create two columns for layout
    col1, col2 = st.columns(2)

    # First column: Nutrition Input and New Meal Entry
    with col1:
        st.write("### Nutrition")

        # Form to input calories consumed manually
        with st.form(key='calories_form'):
            calories = st.number_input("Calories Consumed", min_value=0, step=1)
            submit_calories = st.form_submit_button(label='Add Calories')

            if submit_calories:
                st.success(f"Added {calories} calories")
                load_and_update_final_csv('data/processed/csv/', 'calories_consumed', GIVEN_DATE, calories)

        st.markdown("---")  # Separator

        # New Form: Add Food Item with Meal Type
        with st.form(key='food_entry_form'):
            st.write("### Add Food Item")

            # Step 1: Select Meal Type
            meal_types = [
                'Breakfast',
                'Lunch',
                'Dinner',
                'Snack',          # Translation of 'gouté'
                'Pre-Training',
                'Post-Training',
                'Brunch',
                'Tea Time',
                'Midnight Snack',
                'Other'
            ]
            meal_type = st.selectbox("Select Meal Type", options=meal_types)

            # Step 2: Search for Food Item
            search_query = st.text_input("Search for a food item")

            if search_query:
                # Filter foods_df based on search query (case-insensitive)
                filtered_foods = foods_df[foods_df['food'].str.contains(search_query, case=False, na=False)]
            else:
                filtered_foods = pd.DataFrame()

            # Display search results if any
            if not filtered_foods.empty:
                # Limit to top 10 results for usability
                top_foods = filtered_foods['food'].head(10).tolist()
                selected_food = st.selectbox("Select Food Item", options=top_foods)
            else:
                selected_food = None

            # Step 3: Confirm and Add Food Item
            submit_food = st.form_submit_button(label='Add Food Item')

            if submit_food and selected_food:
                # Retrieve the selected food's caloric value
                caloric_value = foods_df.loc[foods_df['food'] == selected_food, 'Caloric Value'].values[0]

                st.success(f"Added {selected_food} ({caloric_value} kcal) to {meal_type}")

                # Update the calories consumed
                load_and_update_final_csv('data/processed/csv/', 'calories_consumed', GIVEN_DATE, caloric_value)
            elif submit_food and not selected_food:
                st.error("Please select a food item to add.")

    # Second column: Display Calories Information
    with col2:
        st.write("#### Today's Calories")

        # Load the plan_my_day DataFrame
        df = load_and_update_final_csv('data/processed/csv/', "plan_my_day")

        # Retrieve calories consumed today
        calories_data = df.loc[GIVEN_DATE, 'CaloriesConsumed'] if GIVEN_DATE in df.index else 0

        if isinstance(calories_data, pd.Series):
            # If it's a Series, drop NaNs and take the first non-empty value
            calories_consumed = calories_data.dropna().iloc[0] if not calories_data.dropna().empty else 0
        else:
            # If it's a single value (numpy.float64), use it directly (it could be NaN)
            calories_consumed = calories_data if not pd.isna(calories_data) else 0

        # Display calories consumed
        st.write(f"**Calories Consumed Today:** {calories_consumed} kcal")

        # Calculate total daily calorie needs
        total_daily_calories = st.session_state['user_data']['passive_calories'] + st.session_state['user_data']['active_calories']
        calories_remaining = total_daily_calories - calories_consumed
        st.write(f"**Calories To Consume Today:** {calories_remaining} kcal")

        # Display progress bar
        progress = min(calories_consumed / total_daily_calories, 1.0)
        st.progress(progress)

        # Provide guidance on managing remaining calories
        if calories_remaining > 0:
            st.write("🟢 You have room for more food today!")
        else:
            st.write("🔴 You've reached or exceeded your calorie limit for today. Consider balancing your intake.")
