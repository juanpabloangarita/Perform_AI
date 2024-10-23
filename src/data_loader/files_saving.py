# save_final_csv.py

import os
import logging
from .get_full_path import get_full_path

class Sourcer:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    @staticmethod
    def save_final_csv(file_path=None, w_df=None, a_df=None, df=None, foods_df=None):
        full_path = get_full_path(file_path)
        for name, data in zip(['final_df', 'foods_df', 'workouts_df', 'activities_df'], [df, foods_df, w_df, a_df]):
            if data is not None:
                try:
                    data.to_csv(os.path.join(full_path, f"{name}.csv"), index = name in ['final_df', 'foods_df'], na_rep='') # TODO: Check if index = False is good for other dataframes
                    logging.info(f"{name.replace('_', ' ').title()} dataframe saved successfully")
                except Exception as e:
                    logging.error(f"Error saving dataframe: {e}")
