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

from config import setup_paths
from src.main import main
from dashboard_plot import *
from params import *

st.set_page_config(layout="wide")  # Set the layout to wide to utilize more space
# Define your credentials here (use environment variables or a secure method in production)
USERNAME = "juanpa"
PASSWORD = "1234"

# Initialize session state for authentication and username
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

# Function to display the login form
def show_login_form():
    st.subheader('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        if username == USERNAME and password == PASSWORD:
            st.session_state['authenticated'] = True
            st.session_state['username'] = username
            st.success('Login successful!')
            st.session_state['show_home_button'] = True
        else:
            st.session_state['authenticated'] = False
            st.error('Invalid username or password')

# Check if user is authenticated
if not st.session_state['authenticated']:
    show_login_form()
    # Button to go to the main app
    if st.button('Go to the App'):
        st.session_state['show_login'] = False  # Ensure login form is not shown

else:
    # Display main app content if authenticated
    st.title('PERFORM_AI')
    st.title('Your Health AI Assistant')
    st.write(f'Welcome {st.session_state["username"]}!')

    # Option for user to select data source
    data_source = st.radio(
        "Select Data Source",
        ('Use Local Data', 'Upload New Data')
    )

    uploaded_files = None
    if data_source == 'Upload New Data':
        uploaded_files = st.file_uploader(
            label="Please insert your CSV Files",
            type=['csv', 'xlsx'],
            accept_multiple_files=True,
            key="fileUploader"
        )

    if data_source == 'Use Local Data' or uploaded_files:
        if data_source == 'Use Local Data':
            # Call the main function from src/main.py
            tss_df, atl_df, ctl_df, tsb_df, w_df, a_df, final_df = main()
        else:
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
                workouts_df.to_csv(f's3://{BUCKET_NAME}/csv/workout_test_02.csv', index=False)

                # Optionally, re-read the saved file from the S3 bucket (if needed)
                workouts_df = pd.read_csv(f's3://{BUCKET_NAME}/csv/workout_test_02.csv')

                st.write("Files successfully processed and uploaded to S3.")

                # Process the data using the main function
                tss_df, atl_df, ctl_df, tsb_df, w_df, a_df, final_df = main(workouts_df)

                # Display a success message or further processing results
                st.write("Processing completed successfully.")

            except Exception as e:
                st.error(f"An error occurred while processing the files: {e}")

        # Make sure the chart uses the full container width
        # Your Plotly chart code
        fig = plot_dashboard(tss_df, atl_df, ctl_df, tsb_df)

        # Set figure size by updating the layout
        fig.update_layout(
            autosize=True,
            width=1200,  # Adjust width as needed
            height=600   # Adjust height as needed
        )

        st.plotly_chart(fig, use_container_width=True)


        if st.checkbox('Show FINAL DataFrame'):
            st.write(final_df)
        # Option to display w_df
        if st.checkbox('Show Workouts DataFrame'):
            st.write(w_df)
        # Option to display a_df
        if st.checkbox('Show Activities DataFrame'):
            st.write(a_df)
    else:
        st.write("Please upload a file")
