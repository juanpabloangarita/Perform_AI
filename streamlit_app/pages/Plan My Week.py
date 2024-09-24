from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located - pages
dir_script_dir = os.path.dirname(script_dir) #directory = streamlit_app
dir_script_dir = os.path.dirname(dir_script_dir) #src
sys.path.append(dir_script_dir)


from src.data_processing import load_and_update_final_csv
from src.training_peaks import navigate_to_login
from params import *

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
final_df = load_and_update_final_csv('data/processed/csv/', 'home')

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

    # # Today Button
    # with col3:
    #     if st.button("Today"):
    #         st.session_state.week_start = get_monday(current_date)

    # # Navigation Buttons (< and >)
    # with col4:
    #     prev_week, next_week = st.columns([1, 1], gap="small")

    #     with prev_week:
    #         if st.button("<"):
    #             st.session_state.week_start -= timedelta(weeks=1)

    #     with next_week:
    #         if st.button("\>"):
    #             st.session_state.week_start += timedelta(weeks=1)

    with col2:
        if st.button("Training Peaks Reload"):
            tp_data_update = navigate_to_login('both')
            load_and_update_final_csv('data/processed/csv/', "training_peaks", tp_data_update)

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
                        header_text = f"**{row['WorkoutType']}**  \n{row['Title']}  \n{row['PlannedDuration']} hours"
                        with st.expander(header_text, expanded=False):
                            workout_type = st.text_input("Workout Type", row['WorkoutType'], key=f"WorkoutType_{idx}{row}")
                            planned_duration = st.text_input("Time", row['PlannedDuration'], key=f"Time_{idx}{row}")
                            planned_distance = st.number_input("Planned Distance (m)", row['PlannedDistanceInMeters'], key=f"Distance_{idx}{row}")
                            calories_spent = st.number_input("Calories Spent", row['CaloriesSpent'], key=f"Calories_{idx}{row}")

                            # Editable description and comments
                            description = row['WorkoutDescription']
                            coach_comments = row['CoachComments']

                            st.markdown(f"**Description**: {description}")
                            st.markdown(f"**Coach Comments**: {coach_comments}")

                            # Save the modifications to session_state temp dictionary
                            if st.button("Update Training", key=f"update_{idx}{row}"):
                                st.session_state.temp_data[idx] = {
                                    'WorkoutType': workout_type,
                                    'PlannedDuration': planned_duration,
                                    'PlannedDistanceInMeters': planned_distance,
                                    'CaloriesSpent': calories_spent
                                }

                else:
                    st.markdown("<span style='font-size:16px;'>No workout data.</span>", unsafe_allow_html=True)

# Run the main function
main()
