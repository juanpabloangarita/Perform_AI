# home.py

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
import os
import boto3

from src.main import main
#from dashboard_plot import *
from params import *

# Ensure the parent directory is in the Python path
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# sys.path.append(parent_dir)
from config import setup_paths

# Set up the Python path
setup_paths()

# Your code here


st.title('Health AI Assistant')
st.write('Welcome to the Health AI Assistant!')

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
        print("ho ho")
        #tss_df, atl_df, ctl_df, tsb_df, w_df = main()
    else:
        try:
            # List to hold DataFrames
            df_list = []

            # Process each uploaded file
            for uploaded_file in uploaded_files:
                # Read each file into a DataFrame
                df = pd.read_csv(uploaded_file)#, index_col=0) WARNING index_col=0, pandas uses the first column of the CSV file as the index of the DataFrame. If the "Title" column is the first column in the CSV file, it will be used as the index,
                df_list.append(df)

            # Concatenate all DataFrames
            workouts_df = pd.concat(df_list, ignore_index=True)

            # Save the concatenated DataFrame to the specified S3 bucket
            workouts_df.to_csv(f's3://{BUCKET_NAME}/csv/workout_test_02.csv', index=False) # when uploading one file, i didn't have index = false, i suppose this is why i got to put index_col = 0

            # Optionally, re-read the saved file from the S3 bucket (if needed)
            workouts_df = pd.read_csv(f's3://{BUCKET_NAME}/csv/workout_test_02.csv')#, index_col=0)

            st.write("Files successfully processed and uploaded to S3.")

            # Process the data using the main function
            #tss_df, atl_df, ctl_df, tsb_df, w_df = main(workouts_df)

            # Display a success message or further processing results
            st.write("Processing completed successfully.")

        except Exception as e:
            st.error(f"An error occurred while processing the files: {e}")

    # Example usage with Streamlit
    #fig = plot_dashboard(tss_df, atl_df, ctl_df, tsb_df)
    #st.plotly_chart(fig)

    # Option to display w_df
    if st.checkbox('Show DataFrame'):
        st.write('haha')#st.write(w_df)
else:
    st.write("Please upload a file")
