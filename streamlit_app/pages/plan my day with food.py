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


# Ensure that st.session_state['meals'] is initialized
if 'meals' not in st.session_state:
    st.session_state['meals'] = {'Breakfast': [], 'Lunch': [], 'Dinner': [], 'Gouté': []}

with st.container():
    # Create two columns for layout
    col1, col2 = st.columns(2)
    foods_df = load_foods_df()

    # First column: Input calories consumed via food selection
    with col1:
        st.write("### Nutrition")

        with st.form(key='food_form'):
            # Food selection with search capability
            food_search = st.text_input("Search for a food item", "").strip().lower()

            # Filter foods_df based on search input
            if food_search:
                filtered_foods = foods_df[foods_df['food'].str.lower().str.contains(food_search)]
            else:
                filtered_foods = foods_df.copy()  # Ensure it’s always a DataFrame

            # Limit to top 100 results to prevent performance issues
            if not filtered_foods.empty:
                filtered_foods = filtered_foods.head(100)
            else:
                st.warning("No data available.")
                filtered_foods = pd.DataFrame(columns=['food'])  # Create an empty DataFrame to avoid NoneType

            # Selectbox for food items with autocomplete
            food_item = st.selectbox("Select Food", options=filtered_foods['food'].unique() if not filtered_foods.empty else [])

            # Input number of units
            number_of_units = st.number_input("Number of Units", min_value=1, step=1, value=1)

            # Input grams per unit
            grams_per_unit = st.number_input("Grams per Unit", min_value=1.0, step=1.0, value=100.0)

            # Select meal
            meal_options = list(st.session_state['meals'].keys())
            selected_meal = st.selectbox("Select Meal", options=meal_options)

            # Add submit button
            submit_food = st.form_submit_button(label='Add Food')

            if submit_food and food_item:
                # Retrieve the food details from foods_df
                food_details = foods_df[foods_df['food'] == food_item].iloc[0]

                # Calculate total grams
                total_grams = number_of_units * grams_per_unit

                # Calculate scaling factor based on 100g
                scaling_factor = total_grams / 100.0

                # Calculate nutritional values
                nutritional_info = {
                    'Food': food_item,
                    'Units': number_of_units,
                    'Grams per Unit': grams_per_unit,
                    'Total Grams': total_grams,
                    'Calories': food_details['Caloric Value'] * scaling_factor,
                    'Fat': food_details['Fat'] * scaling_factor,
                    'Saturated Fats': food_details['Saturated Fats'] * scaling_factor,
                    'Monounsaturated Fats': food_details['Monounsaturated Fats'] * scaling_factor,
                    'Polyunsaturated Fats': food_details['Polyunsaturated Fats'] * scaling_factor,
                    'Carbohydrates': food_details['Carbohydrates'] * scaling_factor,
                    'Sugars': food_details['Sugars'] * scaling_factor,
                    'Protein': food_details['Protein'] * scaling_factor,
                    'Dietary Fiber': food_details['Dietary Fiber'] * scaling_factor,
                }

                # Append to the selected meal
                st.session_state['meals'][selected_meal].append(nutritional_info)

                st.success(f"Added {number_of_units} x {food_item} ({total_grams}g) to {selected_meal}")

        # Display added foods per meal
        for meal, foods in st.session_state['meals'].items():
            if foods:
                st.write(f"**{meal}**")
                meal_df = pd.DataFrame(foods)
                st.dataframe(meal_df[['Food', 'Units', 'Grams per Unit', 'Total Grams', 'Calories', 'Fat', 'Protein', 'Carbohydrates']])

        # Ensure space is balanced between columns
        with st.empty():
            pass

    # Second column: Display calories and remaining calories
    with col2:
        st.write("#### Today's Nutrition Summary")

        # Initialize totals
        total_calories_consumed = 0
        total_fat = 0
        total_protein = 0
        total_carbs = 0
        total_saturated_fats = 0
        total_monounsaturated_fats = 0
        total_polyunsaturated_fats = 0
        total_sugars = 0
        total_dietary_fiber = 0

        # Calculate totals from meals
        for foods in st.session_state['meals'].values():
            for food in foods:
                total_calories_consumed += food['Calories']
                total_fat += food['Fat']
                total_protein += food['Protein']
                total_carbs += food['Carbohydrates']
                total_saturated_fats += food['Saturated Fats']
                total_monounsaturated_fats += food['Monounsaturated Fats']
                total_polyunsaturated_fats += food['Polyunsaturated Fats']
                total_sugars += food['Sugars']
                total_dietary_fiber += food['Dietary Fiber']

        # Display totals
        st.write(f"**Total Calories Consumed Today:** {total_calories_consumed:.1f} kcal")
        st.write(f"**Total Fat:** {total_fat:.1f} g")
        st.write(f"**Total Protein:** {total_protein:.1f} g")
        st.write(f"**Total Carbohydrates:** {total_carbs:.1f} g")
        st.write(f"**Total Saturated Fats:** {total_saturated_fats:.1f} g")
        st.write(f"**Total Monounsaturated Fats:** {total_monounsaturated_fats:.1f} g")
        st.write(f"**Total Polyunsaturated Fats:** {total_polyunsaturated_fats:.1f} g")
        st.write(f"**Total Sugars:** {total_sugars:.1f} g")
        st.write(f"**Total Dietary Fiber:** {total_dietary_fiber:.1f} g")

        # Assuming you have a variable to hold total daily calories (for example)
        total_daily_calories = 2000  # Replace with your logic to fetch total daily calories
        remaining_calories = total_daily_calories - total_calories_consumed
        st.write(f"**Remaining Calories for Today:** {remaining_calories:.1f} kcal")


# You can add more sections or functionalities below as needed
