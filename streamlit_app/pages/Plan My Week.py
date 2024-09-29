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


from src.data_processing import load_and_update_final_csv
from src.training_peaks import navigate_to_login
from params import *


# Function to stop the EC2 instance if it's running
def stop_ec2_instance(instance_id):
    ec2 = boto3.resource('ec2')
    instance = ec2.Instance(instance_id)
    if instance.state['Name'] == 'running':
        print("Stopping EC2 instance...")
        instance.stop()
        instance.wait_until_stopped()
        print("EC2 instance stopped.")

# Function to start the EC2 instance if it's not running
def trigger_ec2_instance(instance_id):
    ec2 = boto3.resource('ec2')
    instance = ec2.Instance(instance_id)
    if instance.state['Name'] != 'running':
        print("Starting EC2 instance...")
        instance.start()
        instance.wait_until_running()
        print("EC2 instance started.")

""" # NOTE: FOR SSM SERVICE
# Function to run the training_peaks.py script on EC2 via SSM
def run_script_on_ec2(instance_id):
    ssm_client = boto3.client('ssm')

    response = ssm_client.send_command(
        InstanceIds=[instance_id],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': ['python3 /home/ec2-user/Perform_AI/training_peaks.py']}
    )

    # Retrieve the Command ID to track the execution
    command_id = response['Command']['CommandId']

    # Wait for the command to complete successfully
    ssm_client.get_waiter('command_succeeded').wait(CommandId=command_id, InstanceId=instance_id)

    # Retrieve the overall status of the command
    command_status = response['Command']['Status']
    print(f"Script executed on EC2 instance with status: {command_status}")

    # Retrieve the exit code to determine the script's execution outcome
    output = ssm_client.list_command_invocations(CommandId=command_id, Details=True)
    exit_code = output['CommandInvocations'][0]['CommandPlugins'][0]['ResponseCode']

    return command_status, exit_code
"""
# # NOTE: INSTEAD OF PREVIOUS FUNCTION
# Function to run the training_peaks.py script on EC2 via SSH
def run_script_via_ssh(instance_ip):
    try:
        # Retrieve the SSH key from Streamlit secrets
        # ssh_key_str = st.secrets["ssh"]["ssh_key"]
        # Use StringIO to convert the string to a file-like object
        key_file = StringIO(SSH_KEY_STR)
        # Load the private key
        private_key = paramiko.RSAKey.from_private_key(key_file)

        # Create an SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the EC2 instance
        ssh.connect(hostname=instance_ip, username='ec2-user', pkey=private_key)

        # Execute the training_peaks.py script
        stdin, stdout, stderr = ssh.exec_command('python3 /home/ec2-user/Perform_AI/training_peaks.py')

        # Wait for the command to complete
        exit_status = stdout.channel.recv_exit_status()

        # Close the SSH connection
        ssh.close()

        # Determine command_status based on exit_status
        if exit_status == 0:
            command_status = "Success"
        elif exit_status == 1:
            command_status = "Partial Success"
        else:
            command_status = "Failed"

        print(f"Script executed on EC2 instance with status: {command_status}")

        return command_status

    except Exception as e:
        print(f"Error during SSH execution: {e}")
        return "Failed", 2


# Function to fetch the scraped data from S3
def fetch_scraped_data():
    scraped_df = pd.read_csv(f"s3://{BUCKET_NAME}/csv/tp_scraped.csv", na_filter=False)
    return scraped_df

# Function to reset the scraped data CSV in S3 by overwriting it with an empty DataFrame
def reset_scraped_data():
    # Define the structure of your DataFrame based on your scraping logic
    empty_df = pd.DataFrame(columns=[
        'Date', 'compliance_status', 'WorkoutType', 'Title',
        'WorkoutDescription', 'CoachComments', 'duration', 'tss'
    ])
    # Overwrite the existing CSV with the empty DataFrame
    empty_df.to_csv(f"s3://{BUCKET_NAME}/csv/tp_scraped.csv", index=False, na_rep='')
    print("S3 CSV file has been reset for the next run.")

# Main execution block triggered by the Streamlit button
if st.button("Training Peaks Reload"):
    # Step 1: Start EC2 instance if not running
    trigger_ec2_instance(INSTANCE_ID)
    time.sleep(30)  # Wait to ensure the instance is fully started

    # Step 2: Run the training_peaks.py script on EC2 via SSM
    # command_status, exit_code = run_script_on_ec2(INSTANCE_ID)
    command_status = run_script_via_ssh(INSTANCE_ID)

    # Step 3: Stop the EC2 instance to save costs
    stop_ec2_instance(INSTANCE_ID)

    # Step 4: Handle the execution result based on command_status and exit_code
    if command_status == "Success":
        # Step 5: Fetch the scraped data from S3
        scraped_df = fetch_scraped_data()
        st.write(scraped_df)  # Display the DataFrame in Streamlit

        # Step 6: Reset the S3 CSV file for the next run
        reset_scraped_data()

    elif command_status == "Partial Success":
        st.warning("Partial success: Some data was scraped but not all.")
        scraped_df = fetch_scraped_data()
        st.write(scraped_df)

        reset_scraped_data()

    else:
        st.error("Script execution failed; skipping data retrieval.")





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

    with col2:
        if st.button("Training Peaks Reload"):
            # scraped_df = pd.read_csv(f"s3://{BUCKET_NAME}/csv/tp_scraped.csv", na_filter=False) # NOTE: to test if df in s3
            # load_and_update_final_csv('data/processed/csv/', "training_peaks", data_to_update=scraped_df)
            tp_data_update = navigate_to_login('both')
            load_and_update_final_csv('data/processed/csv/', "training_peaks", data_to_update=tp_data_update)

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
                        header_text = f"**{row['WorkoutType']}**  \n{row['Title']}  \n{row['TimeTotalInHours']} hours"
                        with st.expander(header_text, expanded=False):
                            workout_type = st.text_input("Workout Type", value = row['WorkoutType'], key=f"WorkoutType_{idx}{row}")
                            duration = st.text_input("Time", value = row['TimeTotalInHours'], key=f"Time_{idx}{row}")
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

                            # Save the modifications to session_state temp dictionary
                            if st.button("Update", key=f"update_{idx}{row}"):
                                tmp_dict_week = {
                                    'WorkoutType': workout_type,
                                    'TimeTotalInHours': duration,
                                    'DistanceInMeters': distance,
                                    'CaloriesSpent': calories_spent
                                }
                                st.session_state.temp_data[idx] = tmp_dict_week
                                load_and_update_final_csv('data/processed/csv/', "plan_my_week", day_date_str, tmp_dict_week)
                            # if st.button("Delete", key=f"delete_{idx}{row}"):
                            #     load_and_update_final_csv('data/processed/csv/', "plan_my_week")

                else:
                    st.markdown("<span style='font-size:16px;'>No workout data.</span>", unsafe_allow_html=True)

# Run the main function
main()
