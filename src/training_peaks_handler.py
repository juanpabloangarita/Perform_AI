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

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located - src
dir_script_dir = os.path.dirname(script_dir) #directory = PerformAI
sys.path.append(dir_script_dir)

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
