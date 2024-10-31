"""
# Perform_AI.src.user_data.py

This module contains the UserManager class, responsible for user management tasks
such as creating new user accounts, authenticating existing users, and updating
user data. User data is loaded from and saved to an external CSV file via the FileLoader
and FileSaver classes.

Classes:
    - UserManager: Manages user data including loading, creating, and updating user information.
"""

import pandas as pd
import numpy as np
import bcrypt
import os

from src.data_loader.files_saving import FileSaver
from src.data_loader.files_extracting import FileLoader
from params import USER_DATA_FILE


class UserManager:
    """
    A class to manage user data, including user creation, authentication,
    and updating user data.

    Attributes:
        username (str): The username of the user.
        password (str): The password of the user.
        user_data_df (pd.DataFrame): DataFrame containing all user data loaded from an external file.
        user_row (pd.DataFrame): DataFrame row specific to the current user.
        user_data (dict): Dictionary of user information excluding the password.
        user_index (int): Index of the user in the DataFrame.
        kwargs (dict): Additional keyword arguments for updating user data.
        user_exists (bool): Flag indicating if the user already exists in user_data_df.
    """

    def __init__(self, **kwargs):
        """
        Initializes the UserManager class with user credentials and user data.

        Args:
            kwargs: Keyword arguments containing 'username' and 'password'.
        """
        self.username = kwargs.get('username')
        self.password = kwargs.get('password')

        # Load user data from external source
        self.user_data_df = FileLoader().load_user_data()

        # Placeholder attributes
        self.user_row = None
        self.user_data = None
        self.user_index = None

        # Store additional parameters for potential updates
        self.kwargs = kwargs

        # Determine if user already exists in user data
        self.user_exists = self.username in self.user_data_df['username'].values if self.user_data_df is not None else False

    def load_user_data(self, authentication_required=True):
        """
        Loads the data for the specified user and verifies the password if required.

        Args:
            authentication_required (bool): Whether password verification is required.

        Sets:
            user_row, user_data, user_index if user exists and authentication succeeds.
        """
        # Filter user data for the current username
        self.user_row = self.user_data_df[self.user_data_df['username'] == self.username]

        if not self.user_row.empty:
            user_data = self.user_row.iloc[0].to_dict()  # Convert user row to dictionary
            stored_password = user_data['password']

            # Verify password if authentication is required
            if authentication_required:
                if bcrypt.checkpw(self.password.encode('utf-8'), stored_password.encode('utf-8')):
                    # Remove sensitive password info and set user_data attributes
                    del user_data['password']
                    self.user_index = self.user_row.index[0]
                    self.user_data = user_data
            else:
                # Set user_data attributes without password verification
                self.user_index = self.user_row.index[0]
                self.user_data = user_data

    def create_user_data(self):
        """
        Creates a new user with default settings and saves it to the user data DataFrame.

        Notes:
            Default values are used for new user fields like weight, height, age, etc.
            Password is securely hashed using bcrypt.
        """
        # Hash the user's password
        hashed_password = bcrypt.hashpw(self.password.encode('utf-8'), bcrypt.gensalt())

        # Create a DataFrame with new user's data and default values
        new_user_df = pd.DataFrame({
            'username': [self.username],
            'password': [hashed_password.decode('utf-8')],  # Save hashed password
            'weight': [50],  # Default weight
            'height': [50],  # Default height
            'age': [50],     # Default age
            'gender': ['Male'],  # Default gender
            'vo2_max': [50], # Default VO2 max
            'resting_hr': [50], # Default resting heart rate
            'BMR': [0.0], # Changed to 0.0 for consistency and to avoid NaN handling issues
            'goal': ["Lose weight"], # Default goal
            'passive_calories': [0.0] # Changed to 0.0 for consistency and to avoid NaN handling issues
        })

        # Concatenate new user data with existing data or create a new DataFrame
        if self.user_data_df is not None:
            self.user_data_df = pd.concat([self.user_data_df, new_user_df], ignore_index=True)
        else:
            self.user_data_df = new_user_df

        # Save the user data dataframe to a specified file path with the name 'user_data'
        FileSaver().save_dfs(self.user_data_df, file_path=USER_DATA_FILE, name='user_data')

        # Reload user data to refresh the current session data
        self.load_user_data()

    def update_user_data(self):
        """
        Updates the user data based on provided attributes in kwargs, without requiring re-authentication.

        Loads current user data and then applies updates to specified columns
        before saving the changes back to the user data file.

        Notes:
            Only columns that exist in user_data_df will be updated.
        """
        self.load_user_data(authentication_required=False)

        if self.user_data is not None:
            # Iterate through kwargs and update columns if they exist in DataFrame
            for key, value in self.kwargs.items():
                if key in self.user_data_df.columns:
                    self.user_data_df.at[self.user_index, key] = value

            # Save the user data dataframe to a specified file path with the name 'user_data'
            FileSaver().save_dfs(self.user_data_df, file_path=USER_DATA_FILE, name='user_data')
            print(f"User '{self.username}' data updated successfully.")
        else:
            print(f"User '{self.username}' not found in the database.")
