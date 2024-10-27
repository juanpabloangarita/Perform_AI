from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located - pages
dir_script_dir = os.path.dirname(script_dir) #directory = streamlit_app
dir_script_dir = os.path.dirname(dir_script_dir) #src
sys.path.append(dir_script_dir)

from src.data_processing import load_and_update_final_csv
from params import *

# Helper function to get the start of the week (Monday) for a given date
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
final_df = load_and_update_final_csv('home')

# Ensure index is in the proper string format to match
final_df.index = pd.to_datetime(final_df.index).strftime('%Y-%m-%d')

# Main function to handle the calendar view
def main():

    # Set the default date to today
    current_date = datetime.now()

    # Use Streamlit session state to track the currently displayed week
    if 'week_start' not in st.session_state:
        st.session_state.week_start = get_monday(current_date)

    # Create the header with 'Today', '<', '>', and the month/year on the same row, aligned to the left
    col1, _, col3, col4, col5 = st.columns(5, gap="small")  # Left aligned using columns

    with col1:
        st.markdown("<h1 style='margin-bottom: 0;'>Calendar</h1>", unsafe_allow_html=True)
        st.write("")

    # Today Button
    with col3:
        if st.button("Today"):
            st.session_state.week_start = get_monday(current_date)

    # Navigation Buttons (< and >)
    with col4:
        prev_week, next_week = st.columns([1, 1], gap="small")

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

                # Highlight today's date by changing background color
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
                    # Loop through each workout and display the data
                    for idx, row in day_data.iterrows():
                        workout_type = row['WorkoutType']
                        title = row['Title']
                        planned_duration = row['PlannedDuration']
                        calories_spent = row['CaloriesSpent']

                        # Initial view: Show only selected columns
                        with st.container():
                            st.markdown(f"<div style='padding: 8px; border-radius:10px; font-size:14px; border: 1px solid #ddd;'>"
                                        f"<b>{workout_type}</b><br>"
                                        f"<b>{title}</b><br>"
                                        f"<b>Time</b>: {planned_duration}<br>"
                                        f"<b>Calories Spent</b>: {calories_spent}</div>", unsafe_allow_html=True)

                            # Expander for detailed view and modification
                            with st.expander(f"Details for {workout_type} - {title}"):
                                # Display all fields with editable inputs
                                description = st.text_area(
                                    "Workout Description", row['WorkoutDescription'], key=f"description_{idx}"
                                )
                                coach_comments = st.text_area(
                                    "Coach Comments", row['CoachComments'], key=f"coach_comments_{idx}"
                                )
                                planned_distance = st.number_input(
                                    "Planned Distance (meters)", value=row['PlannedDistanceInMeters'], key=f"planned_distance_{idx}"
                                )
                                calories_spent = st.number_input(
                                    "Calories Spent", value=row['CaloriesSpent'], key=f"calories_spent_{idx}"
                                )


                                # Update button
                                if st.button(f"Update info for {title}"):
                                    # Implement logic for updating the data in the dataframe
                                    final_df.at[idx, 'WorkoutDescription'] = description
                                    final_df.at[idx, 'CoachComments'] = coach_comments
                                    final_df.at[idx, 'PlannedDistanceInMeters'] = planned_distance
                                    final_df.at[idx, 'CaloriesSpent'] = calories_spent
                                    st.success("Information updated.")

                            # Add spacing between each activity container
                            st.write("")

                else:
                    st.markdown("<span style='font-size:16px;'>No workout data.</span>", unsafe_allow_html=True)

main()
