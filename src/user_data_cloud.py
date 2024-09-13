# user_data.py

import pandas as pd
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
def save_user_data_cloud(**kwargs):
    username = kwargs['username']
    password = kwargs['password']

    # Hash the password using bcrypt
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Create a DataFrame with the new user's data
    new_user_df = pd.DataFrame({
        'username': [username],
        'password': [hashed_password.decode('utf-8')],  # Save the hashed password
        'weight': kwargs.get('weight', None),
        'height': kwargs.get('height', None),
        'age': kwargs.get('age', None),
        'gender': kwargs.get('gender', None),
        'vo2_max': kwargs.get('vo2_max', None),
        'resting_hr': kwargs.get('resting_hr', None),
        'goal': kwargs.get('goal', None),
        'bmr': kwargs.get('bmr', None),
        'passive_calories': kwargs.get('passive_calories', None)
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


def load_user_data_cloud(username):
    try:
        user_data_df = pd.read_csv(f's3://{BUCKET_NAME}/csv/user_data.csv')
        user_row = user_data_df[user_data_df['username'] == username]
        if not user_row.empty:
            return user_row.iloc[0].to_dict()
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
