# user_data.py

import pandas as pd
import bcrypt
import os
import streamlit as st

from params import *

# Function to check if a user already exists
def check_user_exists(username):
    try:
        user_data_df = pd.read_csv(USER_DATA_FILE)
        return username in user_data_df['username'].values
    except FileNotFoundError:
        return False


#def save_user_data(username, first_name, password, weight, height, age, gender, vo2_max, resting_hr, goal, bmr, passive_calories):
def create_user_data(username, password):
    # Hash the password using bcrypt
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Create a DataFrame with the new user's data
    new_user_df = pd.DataFrame({
        'username': [username],
        'password': [hashed_password.decode('utf-8')],  # Save the hashed password
        'weight': [50],  # Default value for weight
        'height': [50],  # Default value for height
        'age': [50],     # Default value for age
        'gender': ['male'],  # Default value for gender
        'vo2_max': [50], # Default value for VO2 max
        'resting_hr': [50], # Default value for resting heart rate
        'BMR': [None]
    })

    # Read the existing user data
    try:
        user_data_df = pd.read_csv(USER_DATA_FILE)

        # Append the new user's data
        user_data_df = pd.concat([user_data_df, new_user_df], ignore_index=True)
    except FileNotFoundError:
        # Create a new file if it doesn't exist
        user_data_df = new_user_df

    # Save the updated DataFrame to CSV
    user_data_df.to_csv(USER_DATA_FILE, index=False)  # -> data/user_data.csv WHERE DOES IT GO? WARNING


def update_user_data(**kwargs):
    username = kwargs['username']

    # Load existing user data
    user_data_df = pd.read_csv(USER_DATA_FILE)

    # Check if the user exists in the DataFrame based on 'username'
    if username in user_data_df['username'].values:
        # Find the row index for the user
        user_index = user_data_df[user_data_df['username'] == username].index[0]

        # Update the user's data in the DataFrame
        for key, value in kwargs.items():
            if key in user_data_df.columns:
                user_data_df.at[user_index, key] = value  # Update value in the DataFrame

        # Save the updated DataFrame back to CSV
        user_data_df.to_csv(USER_DATA_FILE, index=False)
        print(f"User '{username}' data updated successfully.")

    else:
        print(f"User '{username}' not found in the database.")


def load_user_data(username):
    try:
        user_data_df = pd.read_csv(USER_DATA_FILE)
        user_row = user_data_df[user_data_df['username'] == username]
        if not user_row.empty:
            print()
            print()
            print()
            print(user_row)
            print()
            print()
            print()
            return user_row.iloc[0].to_dict()
        return None
    except FileNotFoundError:
        return None


def authenticate_user(username, password):
    try:
        user_data_df = pd.read_csv(USER_DATA_FILE)
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
