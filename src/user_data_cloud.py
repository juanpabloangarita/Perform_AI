# user_data.py

import pandas as pd
import numpy as np
import bcrypt
import os

from params import *

# Function to check if a user already exists
def check_user_exists_cloud(username):
    try:
        user_data_df = pd.read_csv(f's3://{BUCKET_NAME}/csv/user_data.csv')
        return username in user_data_df['username'].values
    except FileNotFoundError:
        return False


#def save_user_data(username, first_name, password, weight, height, age, gender, vo2_max, resting_hr, goal, bmr, passive_calories):
def create_user_data_cloud(username, password):
    # Hash the password using bcrypt
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Create a DataFrame with the new user's data
    new_user_df = pd.DataFrame({
        'username': [username],
        'password': [hashed_password.decode('utf-8')],  # Save the hashed password
        'weight': [50],  # Default value for weight
        'height': [50],  # Default value for height
        'age': [50],     # Default value for age
        'gender': ['Male'],  # Default value for gender
        'vo2_max': [50], # Default value for VO2 max
        'resting_hr': [50], # Default value for resting heart rate
        'BMR': [np.nan],
        'goal': ["Lose weight"],
        'passive_calories': [np.nan]
    })

    # Read the existing user data
    try:
        user_data_df = pd.read_csv(f's3://{BUCKET_NAME}/csv/user_data.csv')

        # Append the new user's data
        user_data_df = pd.concat([user_data_df, new_user_df], ignore_index=True)
    except FileNotFoundError:
        # Create a new file if it doesn't exist
        user_data_df = new_user_df

    # Save the updated DataFrame to CSV
    user_data_df.to_csv(f's3://{BUCKET_NAME}/csv/user_data.csv', index=False)


def update_user_data_cloud(**kwargs):
    username = kwargs['username']

    # Load existing user data
    user_data_df = pd.read_csv(f's3://{BUCKET_NAME}/csv/user_data.csv')

    # Check if the user exists in the DataFrame based on 'username'
    if username in user_data_df['username'].values:
        # Find the row index for the user
        user_index = user_data_df[user_data_df['username'] == username].index[0]

        # Update the user's data in the DataFrame
        for key, value in kwargs.items():
            if key in user_data_df.columns:
                user_data_df.at[user_index, key] = value  # Update value in the DataFrame

        # Save the updated DataFrame back to CSV
        user_data_df.to_csv(f's3://{BUCKET_NAME}/csv/user_data.csv', index=False)
        print(f"User '{username}' data updated successfully.")

    else:
        print(f"User '{username}' not found in the database.")


def load_user_data_cloud(username):
    try:
        user_data_df = pd.read_csv(f's3://{BUCKET_NAME}/csv/user_data.csv')
        user_row = user_data_df[user_data_df['username'] == username]
        if not user_row.empty:
            user_data = user_row.iloc[0].to_dict()

            # Remove the password from the dictionary for security reasons
            if 'password' in user_data:
                del user_data['password']

            return user_data
        return None
    except FileNotFoundError:
        return None


def authenticate_user_cloud(username, password):
    try:
        user_data_df = pd.read_csv(f's3://{BUCKET_NAME}/csv/user_data.csv')
        user_row = user_data_df[user_data_df['username'] == username]

        if user_row.empty:
            return False

        stored_password = user_row['password'].values[0]
        # Compare the hashed password
        if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
            return True
        return False
    except FileNotFoundError:
        return False
