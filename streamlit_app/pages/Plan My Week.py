# Perform_AI.streamlit_app.plan my week.py

from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import os
import sys
import boto3
import subprocess
import time
from io import StringIO
import paramiko

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located - pages
dir_script_dir = os.path.dirname(script_dir) #directory = streamlit_app
dir_script_dir = os.path.dirname(dir_script_dir) #src
sys.path.append(dir_script_dir)


from src.training_peaks import navigate_to_login
from src.training_peaks_handler import *
from params import *
from src.calorie_estimation_models import load_model
from params import BEST_MODEL, GIVEN_DATE
from src.data_loader.files_extracting import FileLoader


headless_mode = (CLOUD_ON == 'yes')


# Helper function to get the start of the week (Monday)
def get_monday(d: datetime):
    return d - timedelta(days=d.weekday())

# Function to get the list of dates for the week (Monday to Sunday)
def get_week_dates(week_start):
    return [week_start + timedelta(days=i) for i in range(7)]

# Function to highlight the current day
def highlight_today(week_dates):
    today = datetime.now().date()
    return [date.date() == today for date in week_dates]

# Load and update the dataframe
final_df = FileLoader().update_final_df('home')

# Ensure index is in the proper string format to match
final_df.index = pd.to_datetime(final_df.index).strftime('%Y-%m-%d')

# Main function to handle the calendar view
def main():
    # Set the default date to today
    current_date = datetime.now()

    # Use Streamlit session state to track the currently displayed week
    if 'week_start' not in st.session_state:
        st.session_state.week_start = get_monday(current_date)

    if 'temp_data' not in st.session_state:
        st.session_state.temp_data = {}

    # Create the header with 'Today', '<', '>', and the month/year on the same row, aligned to the left
    col1, col2, col3, col4, col5 = st.columns(5, gap="small")  # Left aligned using columns

    with col1:
        st.markdown("<h1 style='margin-bottom: 0;'>Calendar</h1>", unsafe_allow_html=True)

    with col2:
        if st.button("Training Peaks Reload"):
            if headless_mode:
                command_status = training_peaks_button_helper()
                # Step 4: Handle the execution result based on command_status and exit_code
                if command_status == "Success":
                    # Step 5: Fetch the scraped data from S3
                    scraped_df = fetch_scraped_data()
                    st.write(scraped_df)  # Display the DataFrame in Streamlit
                    FileLoader().update_final_df("training_peaks", data_to_update=scraped_df)
                    reset_scraped_data()

                elif command_status == "Partial Success":
                    st.warning("Partial success: Some data was scraped but not all.")
                    scraped_df = fetch_scraped_data()
                    st.write(scraped_df)
                    FileLoader().update_final_df("training_peaks", data_to_update=scraped_df)
                    reset_scraped_data()

                else:
                    st.error("Script execution failed; skipping data retrieval.")
            else:
                tp_data_update = navigate_to_login('both')
                FileLoader().update_final_df("training_peaks", data_to_update=tp_data_update)

    with col3:
        st.write("")

    # Navigation Buttons (< and >)
    with col4:
        button_1, prev_week, next_week = st.columns([1, 1, 1], gap="small")
        with button_1:
            if st.button("Today"):
                st.session_state.week_start = get_monday(current_date)
        with prev_week:
            if st.button("<"):
                st.session_state.week_start -= timedelta(weeks=1)
        with next_week:
            if st.button("\>"):
                st.session_state.week_start += timedelta(weeks=1)


    # Current Month and Year
    with col5:
        current_month_year = st.session_state.week_start.strftime("%B %Y")
        st.markdown(f"<h4 style='text-align:left; margin-bottom: 0;'>{current_month_year}</h4>", unsafe_allow_html=True)

    # Get the current week's dates (Monday to Sunday)
    week_dates = get_week_dates(st.session_state.week_start)

    # Determine which day is today to highlight
    today_highlight = highlight_today(week_dates)

    # Display the week view using columns for each day
    st.divider()  # Separate navigation from the calendar
    cols = st.columns(7, gap="small")

    for i, col in enumerate(cols):
        with col:
            with st.container():
                # Highlight today's date
                if today_highlight[i]:
                    st.markdown(f"<div style='background-color:#f0f8ff; padding: 10px; border-radius:10px;'>"
                                f"<b>{week_dates[i].strftime('%A')}</b><br><span style='font-size: 18px;'>{week_dates[i].day}</span>"
                                f"</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='padding: 10px; border-radius:10px;'>"
                                f"<b>{week_dates[i].strftime('%A')}</b><br><span style='font-size: 18px;'>{week_dates[i].day}</span>"
                                f"</div>", unsafe_allow_html=True)

                # Filter final_df by the current day
                day_date_str = week_dates[i].strftime('%Y-%m-%d')
                day_data = final_df[final_df.index == day_date_str]

                if not day_data.empty:
                    # Loop through each workout and display the main columns (WorkoutType, Title, Time)
                    for idx, row in day_data.iterrows():
                        header_text = f"**{row['WorkoutType']}**  \n{row['Title']}  \n{round(row['TimeTotalInHours']*60)} minutes"
                        with st.expander(header_text, expanded=False):
                            workout_type = st.text_input("Workout Type", value = row['WorkoutType'], key=f"WorkoutType_{idx}{row}")
                            duration = st.text_input("Time", value = round(row['TimeTotalInHours']*60), key=f"Time_{idx}{row}")
                            distance = st.number_input(
                                "Distance (m)",
                                value=row['DistanceInMeters'],  # Default value
                                min_value=0.0,  # Set minimum value to 0 or any other lower limit you prefer
                                max_value=None,  # No maximum limit
                                key=f"Distance_{idx}{row}"
                            )
                            calories_spent = st.number_input("Calories Spent", value = row['CaloriesSpent'], key=f"Calories_{idx}{row}")

                            # Editable description and comments
                            description = row['WorkoutDescription']
                            coach_comments = row['CoachComments']

                            st.markdown(f"**Description**: {description}")
                            st.markdown(f"**Coach Comments**: {coach_comments}")
                            duration = float(duration)

                            # Save the modifications to session_state temp dictionary
                            if st.button("Update", key=f"update_{idx}{row}"):
                                if duration > 0:
                                    try:
                                        # Load the preprocessing pipeline
                                        preprocessing_pipeline = load_model('preprocessing_pipeline')
                                        if preprocessing_pipeline is None:
                                            st.error("Preprocessing pipeline not found. Please train the model first.")
                                            st.stop()

                                        # Create a DataFrame with required features
                                        input_data = pd.DataFrame({
                                            'WorkoutType': [workout_type],
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

                                tmp_dict_week = {
                                    'WorkoutType': workout_type,
                                    'TimeTotalInHours': duration,
                                    'DistanceInMeters': distance,
                                    'CaloriesSpent': calories_spent,
                                    'estimated_calories': estimated_calories
                                }
                                st.session_state.temp_data[idx] = tmp_dict_week
                                FileLoader().update_final_df("plan_my_week", day_date_str, tmp_dict_week)
                            # if st.button("Delete", key=f"delete_{idx}{row}"):
                            #     FileLoader().update_final_df( "plan_my_week")

                else:
                    st.markdown("<span style='font-size:16px;'>No workout data.</span>", unsafe_allow_html=True)

# Run the main function
main()
