# user_data.py

import pandas as pd
import numpy as np
import bcrypt
import os

from src.data_loader.files_saving import FileSaver
from src.data_loader.files_extracting import FileLoader


class UserManager:
    def __init__(self, **kwargs):
        self.username = kwargs.get('username')
        self.password = kwargs.get('password')
        self.user_data_df = FileLoader().load_user_data()
        self.user_row = None
        self.user_data = None
        self.user_index = None
        self.kwargs = kwargs
        self.user_exists = self.username in self.user_data_df.values

    def load_user_data(self, authentication_required=True):
        self.user_row = self.user_data_df[self.user_data_df['username'] == self.username]
        if not self.user_row.empty:
            user_data = self.user_row.iloc[0].to_dict()
            stored_password = user_data['password']
            # Compare the hashed password
            if authentication_required:
                if bcrypt.checkpw(self.password.encode('utf-8'), stored_password.encode('utf-8')):
                    del user_data['password']
                    self.user_index = self.user_row.index[0]
                    self.user_data = user_data
            else:
                self.user_index = self.user_row.index[0]
                self.user_data = user_data


    def create_user_data(self):
        hashed_password = bcrypt.hashpw(self.password.encode('utf-8'), bcrypt.gensalt())
        # Create a DataFrame with the new user's data
        new_user_df = pd.DataFrame({
            'username': [self.username],
            'password': [hashed_password.decode('utf-8')],  # Save the hashed password
            'weight': [50],  # Default value for weight
            'height': [50],  # Default value for height
            'age': [50],     # Default value for age
            'gender': ['Male'],  # Default value for gender
            'vo2_max': [50], # Default value for VO2 max
            'resting_hr': [50], # Default value for resting heart rate
            'BMR': [np.nan], # FIXME: and below, probably 0.0 instead of np.nan
            'goal': ["Lose weight"],
            'passive_calories': [np.nan] # FIXME
        })

        if self.user_data_df is not None:
            self.user_data_df = pd.concat([self.user_data_df, new_user_df], ignore_index=True)
        else:
            self.user_data_df = new_user_df

        # Save the updated DataFrame to CSV
        FileSaver().save_user_data(self.user_data_df)
        self.load_user_data()

    def update_user_data(self): # HUMAN SETTING
        self.load_user_data(authentication_required=False)
        if self.user_data is not None:
            # Update the user's data in the DataFrame
            for key, value in self.kwargs.items():
                if key in self.user_data_df.columns:
                    self.user_data_df.at[self.user_index, key] = value  # Update value in the DataFrame

            # Save the updated DataFrame back to CSV
            FileSaver().save_user_data(self.user_data_df)
            print(f"User '{self.username}' data updated successfully.")

        else:
            print(f"User '{self.username}' not found in the database.")
