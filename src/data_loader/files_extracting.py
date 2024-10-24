# Perform_AI.src.data_loader.files_extracting.py

import os
import logging
import pandas as pd
from .get_full_path import get_full_path
from params import CLOUD_ON, BUCKET_NAME, USER_DATA_FILE


class Sourcer:
    def __init__(self):
        pass

    @staticmethod
    def load_user_data():
        if CLOUD_ON=='yes':
            user_data = pd.read_csv(f's3://{BUCKET_NAME}/csv/user_data.csv')
        else:
            full_path = get_full_path(USER_DATA_FILE)
            user_data = pd.read_csv(full_path)

        return user_data
