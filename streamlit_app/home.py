# home.py

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
import os
import boto3

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

from config import setup_paths # TODO: decide to erase it or to implement it

from dashboard_plot import *
from params import *

from src.main import main
from src.user_data import *
from src.user_data_cloud import *
from src.data_processing import load_tss_values_for_dashboard, load_and_update_final_csv



st.set_page_config(page_title="Perform AI", page_icon="🌞", layout="wide", initial_sidebar_state="expanded")



# Capture the 'main' argument if provided
main_arg = None
if len(sys.argv) > 1:
    main_arg = sys.argv[1]
if CLOUD_ON == 'yes':
    main_arg = 'main'

# Initialize session state for authentication and username
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

# Function to display the login and sign-up form
def show_login_form():
    st.subheader('Login / Sign Up')
    # Option to switch between login and sign up
    option = st.radio("Select Option", ("Login", "Sign Up"))
    with st.container(border=True):
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')

        if option == "Sign Up":
            secret_code = st.text_input('Secret Code', type='password')
            if st.button('Sign Up'):
                if secret_code == CODE_PROMO:
                    if not check_user_exists(username):
                        create_user_data(username, password)
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = username
                        st.session_state['user_data']= load_user_data(username)
                        response_main = main(st.session_state['user_data'], main_arg=str(main_arg))
                        st.success(f"Sign up successful! {response_main}")
                    else:
                        st.error('Username already exists.')
                else:
                    st.error('Invalid secret code.')
        else:  # Login
            if st.button('Login'):
                if authenticate_user(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.session_state['user_data']= load_user_data(username)
                    response_main = main(st.session_state['user_data'], main_arg=str(main_arg))
                    st.success(f"Login successful! {response_main}")
                else:
                    st.session_state['authenticated'] = False
                    st.error('Invalid username or password')

# Function to display the login and sign-up form
def show_login_form_cloud():
    st.subheader('Login / Sign Up')
    # Option to switch between lofgin and sign up
    option = st.radio("Select Option", ("Login", "Sign Up"))
    with st.container(border=True):
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')

        if option == "Sign Up":
            secret_code = st.text_input('Secret Code', type='password')
            if st.button('Sign Up'):
                if secret_code == CODE_PROMO:
                    if not check_user_exists_cloud(username):
                        create_user_data_cloud(username, password)
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = username
                        st.session_state['user_data']= load_user_data(username)
                        response_main = main(st.session_state['user_data'], main_arg=str(main_arg))
                        st.success(f"Sign up successful! {response_main}")
                    else:
                        st.error('Username already exists.')
                else:
                    st.error('Invalid secret code.')
        else:  # Login
            if st.button('Login'):
                if authenticate_user_cloud(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.session_state['user_data']= load_user_data_cloud(username)
                    response_main = main(st.session_state['user_data'], main_arg=str(main_arg))
                    st.success(f"Login successful! {response_main}")
                else:
                    st.session_state['authenticated'] = False
                    st.error('Invalid username or password')


# Check if user is authenticated
if not st.session_state['authenticated']:
    if CLOUD_ON == 'yes':
        show_login_form_cloud()
    else:
        show_login_form()

    if st.button('GAME ON!'):
        pass # NOTE: weird behaviour, the way streamlit manages the state, it seems that as soon as this button is clicked, nothing below this line is read
else:
    # Display main app content if authenticated
    st.title('PERFORM_AI')
    st.markdown("<h3 style='margin-top: -20px;'>Your Health AI Assistant</h3>", unsafe_allow_html=True)
    st.write("")
    with st.container(border=True):
        # Display user information in one row
        st.write(f'### Welcome {st.session_state["username"]}!')

        # Create a row with four columns
        col1, col2, col3, col4 = st.columns(4)

        # Display the information in each column
        with col1:
            st.write(f"**Sports Scheduled Today:** Yes.")

        with col2:
            st.write(f"**Goal:** {st.session_state['user_data']['goal']}")

        with col3:
            st.write(f"**Maintenance Calories:** {st.session_state['user_data']['passive_calories']} kcal")
        with col4:
            st.write(f"**Meals Pending:** Yes")

    st.write("")
    # Option for user to select data source
    data_source = st.radio(
        f"Select Data Source for your Workouts {st.session_state['username']}",
        ('Use Pre-Saved Data', 'Upload New Data')
    )

    uploaded_files = None
    if data_source == 'Upload New Data': # NOTE: For the moment, i have to upload all files simultaneously, i don't know how to handle loading one file, and then the next.
        uploaded_files = st.file_uploader(
            label="Please insert your CSV Files",
            type=['csv', 'xlsx'],
            accept_multiple_files=True,
            key="fileUploader"
        )

    if data_source == 'Use Pre-Saved Data' or uploaded_files:
        if data_source != 'Use Pre-Saved Data':
            # Call the main function from src/main.py
            try:
                # List to hold DataFrames
                df_list = []

                # Process each uploaded file
                for uploaded_file in uploaded_files:
                    # Read each file into a DataFrame
                    df = pd.read_csv(uploaded_file)
                    df_list.append(df)

                # Concatenate all DataFrames
                workouts_df = pd.concat(df_list, ignore_index=False)

                # Save the concatenated DataFrame to the specified S3 bucket
                workouts_df.to_csv(f"s3://{BUCKET_NAME}/csv/{st.session_state['username']}workouts.csv", index=False)

                # Optionally, re-read the saved file from the S3 bucket (if needed)
                workouts_df = pd.read_csv(f"s3://{BUCKET_NAME}/csv/{st.session_state['username']}workouts.csv")

                st.write("Files successfully processed and uploaded to S3.")

                # Process the data using the main function
                response_main = main(st.session_state['user_data'], workouts_df)

                # Display a success message or further processing results
                st.write(f"{response_main} Processing completed successfully.")

            except Exception as e:
                st.error(f"An error occurred while processing the files: {e}")
        st.write("")
        st.write("### Performance Training")
        if st.checkbox("Show Dashboard"):
            # Load data
            tss_df, atl_df, ctl_df, tsb_df = load_tss_values_for_dashboard('data/processed/csv/')

            # Ensure index is in datetime format
            tss_df.index = pd.to_datetime(tss_df.index)

            st.write("##### Select Date Range to Filter Dashboard")

            # Date selection widgets with default values
            col1, col2 = st.columns(2)

            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=tss_df.index.min().date(),  # Default to the earliest date in tss_df
                    min_value=tss_df.index.min().date(),
                    max_value=tss_df.index.max().date()
                )

            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=tss_df.index.max().date(),  # Default to the latest date in tss_df
                    min_value=start_date,  # The minimum value for end_date is start_date
                    max_value=tss_df.index.max().date()
                )

            # Convert the selected dates back to the '%Y-%m-%d' format for plot_dashboard
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

            # Pass selected start_date and end_date (in string format) to the plot_dashboard function
            fig = plot_dashboard(tss_df, atl_df, ctl_df, tsb_df, start_date_str, end_date_str)

            # Set figure size by updating the layout
            fig.update_layout(
                autosize=True,
                width=1200,  # Adjust width as needed
                height=600   # Adjust height as needed
            )

            # Plot the dashboard
            st.plotly_chart(fig, use_container_width=True)

        if st.checkbox('Show Data'):
            final_df = load_and_update_final_csv('data/processed/csv/', 'home')
            st.write(final_df)
    else:
        st.write("Please upload a file")
