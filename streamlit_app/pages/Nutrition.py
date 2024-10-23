


from src.data_processing import load_and_update_final_csv, load_foods_df, get_full_path
from src.calorie_estimation_models import load_model
from params import BEST_MODEL, GIVEN_DATE
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

from datetime import datetime
from PIL import Image
import cv2

# Add the 'src' directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located - pages
dir_script_dir = os.path.dirname(script_dir)  # directory = streamlit_app
dir_script_dir = os.path.dirname(dir_script_dir)  # src
sys.path.append(dir_script_dir)

from src.data_processing import load_and_update_final_csv, load_foods_df, get_full_path
from src.calorie_estimation_models import load_model
from params import BEST_MODEL, GIVEN_DATE

import requests
import pandas as pd
other_keys = {'status':1, 'status_verbose':'product found'}
import re

def extract_quantity(quantity):
    # Check if the quantity matches the pattern for result A
    match = re.search(r'\(\s*(\d+)\s*x', quantity)
    if match:
        return int(match.group(1))  # Return the captured number (e.g., 2)
    else:
        return 1  # Return 1 for other cases (e.g., result B)


def get_product_info(barcode):
    # Define the API endpoint
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"

    # Make the GET request to the API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON data
        data = response.json()
        # Check if the product exists in the response
        if 'product' in data:
            product_data = data['product']
            # Convert the product data to a DataFrame
            df = pd.DataFrame([product_data])
            return df
        else:
            print("Product not found.")
            return None
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None


def initialize_food_log(file_path):
    """Create an empty dataframe with the required columns and save it as a CSV if not exists."""
    full_path = get_full_path(file_path)
    csv_file_path = os.path.join(full_path, 'user_nutrition.csv')

    columns = ['Timestamp', 'Meal', 'Food', 'Units', 'Grams per Unit', 'Total Grams', 'Calories', 'Fat', 'Saturated Fats',
               'Monounsaturated Fats', 'Polyunsaturated Fats', 'Carbohydrates', 'Sugars', 'Protein', 'Dietary Fiber']

    if not os.path.exists(csv_file_path):
        # Create an empty DataFrame with the necessary columns and save it to a CSV file
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file_path, index=False)
    else:
        # Load the existing CSV
        df = pd.read_csv(csv_file_path)
    return df

def update_food_log(file_path, meal, nutritional_info):
    """Update the food log CSV with the new meal data and a timestamp."""
    df = initialize_food_log(file_path)

    nutritional_info['Timestamp'] = GIVEN_DATE
    nutritional_info['Meal'] = meal

    # Convert the dictionary into a DataFrame by wrapping it in a list
    nutritional_info_df = pd.DataFrame([nutritional_info])

    # Append the new entry to the DataFrame
    # Reindex nutritional_info_df to have the same columns as df, filling missing columns with zeros
    nutritional_info_df_reindexed = nutritional_info_df.reindex(columns=df.columns, fill_value=0)

    # Concatenate the two DataFrames
    df = pd.concat([df, nutritional_info_df_reindexed], axis=0)

    full_path = get_full_path(file_path)
    csv_file_path = os.path.join(full_path, 'user_nutrition.csv')

    # Save the updated DataFrame back to the CSV
    df.to_csv(csv_file_path, index=False)

def load_food_log(file_path):
    """Load the food log from the CSV and return the dataframe."""
    full_path = get_full_path(file_path)
    csv_file_path = os.path.join(full_path, 'user_nutrition.csv')
    if os.path.exists(csv_file_path):
        return pd.read_csv(csv_file_path)
    else:
        return pd.DataFrame()



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



with st.container():
    col1, col2 = st.columns(2)

    # First column: Input calories consumed via food selection
    with col1:
        st.write("### Nutrition")

        # Multiple choice for selecting input method
        input_method = st.radio("Choose how to input food information:",
                                 ("Type Food Item", "Scan Barcode"))
        meal_options = ['Breakfast', 'Lunch', 'Dinner', 'Snack', 'Brunch', 'Pre-Training', 'Post-Training', 'During-Training']
        if input_method == "Type Food Item":
            # Food selection with search capability
            food_search = st.text_input("Search for a food item", "").strip().lower()
            filtered_foods = pd.DataFrame(columns=['food'])  # Create an empty DataFrame to avoid NoneType

            if food_search:
                foods_df = load_foods_df()
                filtered_foods = foods_df[foods_df['food'].str.lower().str.contains(food_search)]
                if not filtered_foods.empty:
                    filtered_foods = filtered_foods.head(100)
                else:
                    st.warning("No data available.")
            else:
                st.warning("Type a food item in the search bar")

            with st.form(key='food_form'):
                # Selectbox for food items with autocomplete
                food_item = st.selectbox("Select Food", options=filtered_foods['food'].unique() if not filtered_foods.empty else [])

                # Input number of units
                number_of_units = st.number_input("Number of Units", min_value=1, step=1, value=1)

                # Input grams per unit
                grams_per_unit = st.number_input("Grams per Unit", min_value=1.0, step=1.0, value=100.0)

                # Select meal
                # meal_options = ['Breakfast', 'Lunch', 'Dinner', 'Snack', 'Brunch', 'Pre-Training', 'Post-Training', 'During-Training']
                selected_meal = st.selectbox("Select Meal", options=meal_options)

                # Add submit button
                submit_food = st.form_submit_button(label='Add Food')

                if submit_food:
                    if food_item:
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

                        # Update the CSV with the new food item and timestamp
                        update_food_log('data/processed/csv/', selected_meal, nutritional_info)

                        st.success(f"Added {number_of_units} x {food_item} ({total_grams}g) to {selected_meal}")
                    else:
                        st.warning("You need to select a food item")

        elif input_method == "Scan Barcode":
            st.write("### Scan Barcode")
            # Capture image using Streamlit
            image = st.camera_input("Capture Barcode Image")

            if image is not None:
                # Open the image from Streamlit as a PIL Image
                img_pil = Image.open(image)

                # Convert the PIL Image to a NumPy array
                img = np.array(img_pil)

                # Convert RGB to BGR format for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Initialize OpenCV Barcode Detector
                bd = cv2.barcode.BarcodeDetector()

                # Detect and decode barcodes
                decoded_info, decoded_type, points = bd.detectAndDecode(img_bgr)

                # Check if barcodes were detected
                if decoded_info:  # Check if there is any decoded info
                    st.success("Barcode Detected!")
                    st.write(f"Decoded Data: {decoded_info}")  # This is the actual barcode data
                    st.write(f"Decoded Type: {decoded_type}")
                    barcode = decoded_info

                    # Get product info
                    product_df = get_product_info(barcode)

                    if not product_df.empty:
                        # Define the meal options
                        meal_options = ['Breakfast', 'Lunch', 'Dinner', 'Snack', 'Brunch', 'Pre-Training', 'Post-Training', 'During-Training']
                        selected_meal = st.selectbox("Select Meal", options=meal_options)

                        # Input number of units
                        number_of_units = st.number_input("Number of Units", min_value=1, step=1, value=1)

                        # Calculate grams per unit based on product info
                        product_df['adjusted_quantity'] = product_df['quantity'].apply(extract_quantity)

                        # Ensure that 'product_quantity' and 'adjusted_quantity' are numeric types
                        product_df['product_quantity'] = pd.to_numeric(product_df['product_quantity'], errors='coerce')
                        product_df['adjusted_quantity'] = pd.to_numeric(product_df['adjusted_quantity'], errors='coerce')

                        grams_per_unit = product_df['product_quantity'].values[0] / product_df['adjusted_quantity'].values[0]


                        # Calculate total grams
                        total_grams = number_of_units * grams_per_unit

                        # Calculate scaling factor based on 100g
                        scaling_factor = total_grams / 100.0

                        # Nutritional info calculation
                        nutritional_info = {
                            'Food': product_df['product_name'].values[0],  # Extracting scalar value
                            'Units': number_of_units,
                            'Grams per Unit': grams_per_unit,  # Extracting scalar value
                            'Total Grams': total_grams,  # Extracting scalar value
                            'Calories': float(product_df['nutriments'][0]['energy-kcal_100g']) * scaling_factor,
                            'Fat': float(product_df['nutriments'][0]['fat_100g']) * scaling_factor,
                            'Carbohydrates': float(product_df['nutriments'][0]['carbohydrates_100g']) * scaling_factor,
                            'Sugars': float(product_df['nutriments'][0]['sugars_100g']) * scaling_factor,
                            'Protein': float(product_df['nutriments'][0]['proteins_100g']) * scaling_factor
                        }



                        # Add submit button for barcode input
                        if st.button('Add Food'):
                            # Update the CSV with the new food item and timestamp
                            update_food_log('data/processed/csv/', selected_meal, nutritional_info)
                            st.success(f"Added {number_of_units} x {nutritional_info['Food']} ({total_grams}g) to {selected_meal}")

                else:
                    st.error("No barcode detected.")

        # Display added foods per meal
        food_log_df = load_food_log('data/processed/csv/')
        food_log_mask = food_log_df['Timestamp']==GIVEN_DATE
        if not food_log_df[food_log_mask].empty:
            for meal in meal_options:
                meal_df = food_log_df[food_log_mask & (food_log_df['Meal'] == meal)]
                if not meal_df.empty:
                    st.write(f"**{meal}**")
                    st.dataframe(meal_df[['Food', 'Units', 'Grams per Unit', 'Total Grams', 'Calories', 'Fat', 'Protein', 'Carbohydrates']])




    # Second column: Display today's nutrition summary
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

        # Calculate totals from the food log
        if not food_log_df[food_log_mask].empty:
            for index, food in food_log_df[food_log_mask].iterrows():
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




        st.write("#### Today's Calories")
        # df = load_and_update_final_csv('data/processed/csv/', "plan_my_day")


        # calories_data = df.loc[GIVEN_DATE, 'CaloriesConsumed'] if GIVEN_DATE in df.index else 0

        # if isinstance(calories_data, pd.Series):
        #     # If it's a Series, drop NaNs and take the first non-empty value
        #     calories_consumed = calories_data.dropna().iloc[0] if not calories_data.dropna().empty else 0
        # else:
        #     # If it's a single value (numpy.float64), use it directly (it could be NaN)
        #     calories_consumed = calories_data if not pd.isna(calories_data) else 0

        # Display calories consumed
        st.write(f"**Calories Consumed Today:** {total_calories_consumed} kcal")

        # Calculate total daily calorie needs
        total_daily_calories = st.session_state['user_data']['passive_calories'] + total_active_calories # NOTE: NOT SURE ABOUT total_active_calories that was created before
        calories_remaining = total_daily_calories - total_calories_consumed
        st.write(f"**Calories Left To Consume Today:** {calories_remaining} kcal")

        # FIXME: Calculations wrong, above and below
        st.progress(min(total_calories_consumed / total_daily_calories, 1.0))  # To ensure progress stays between 0 and 1

        # Provide guidance on managing remaining calories
        if calories_remaining > 0:
            st.write("🟢 You have room for more food today!")
            st.session_state['user_data']['meal'] = 'Yes'
        else:
            st.write("🔴 You've reached or exceeded your calorie limit for today. Consider balancing your intake.")
            st.session_state['user_data']['meal'] = 'No'

        # Ensure space is balanced with the first column
        with st.empty():
            pass
