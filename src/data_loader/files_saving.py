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
                    logging.error(f"Error saving dataframe: {e}")


    def save_tss_values_for_dashboard(self, tss, atl, ctl, tsb, file_path=None):
        full_path = get_full_path(self.file_path if file_path is None else file_path)
        tss.to_csv(os.path.join(full_path, 'tss.csv'), index=True)
        ctl.to_csv(os.path.join(full_path, 'ctl.csv'), index=True)
        atl.to_csv(os.path.join(full_path, 'atl.csv'), index=True)
        tsb.to_csv(os.path.join(full_path, 'tsb.csv'), index=True)
