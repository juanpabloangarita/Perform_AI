# save_final_csv.py

import os
import logging
from .get_full_path import get_full_path

class Sourcer:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.file_path = 'data/processed/csv/'

    def save_final_csv(self, w_df=None, a_df=None, df=None, foods_df=None, file_path=None):
        full_path = get_full_path(self.file_path if file_path is None else file_path)
        for name, data in zip(['final_df', 'foods_df', 'workouts_df', 'activities_df'], [df, foods_df, w_df, a_df]):
            if data is not None:
                try:
                    data.to_csv(os.path.join(full_path, f"{name}.csv"), index = name in ['final_df', 'foods_df'], na_rep='') # TODO: Check if index = False is good for other dataframes
                    logging.info(f"{name.replace('_', ' ').title()} dataframe saved successfully")
                except Exception as e:
                    logging.error(f"Error saving dataframe{name}: {e}")


    def save_tss_values_for_dashboard(self, tss, atl, ctl, tsb, file_path=None):
        full_path = get_full_path(self.file_path if file_path is None else file_path)
        for name, data in zip(['tss', 'atl', 'ctl', 'tsb'],[tss, atl, ctl, tsb]):
            if data is not None:
                try:
                    data.to_csv(os.path.join(full_path, f"{name}.csv"), index=True)
                    logging.info(f"{name.title()} dataframe saved successfully")
                except Exception as e:
                    logging.error(f"Error saving dataframe {name}: {e}")


    def save_during_process(self, work_df, acti_df, file_path=None):
        full_path = get_full_path(self.file_path if file_path is None else file_path)
        work_df.to_csv(os.path.join(full_path, 'workouts_to_process_df.csv'), na_rep='')
        acti_df.to_csv(os.path.join(full_path, 'activities_to_process_df.csv'), na_rep='')
