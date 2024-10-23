# Perform_AI.src.data_loader.files_saving.py

import os
import logging
from .get_full_path import get_full_path

class FileSaver:
    """
    A class responsible for saving various dataframes to CSV files.
    Provides methods for saving workout data, activities, TSS metrics, and nutrition information.
    """

    def __init__(self):
        """Initialize the FileSaver with the default file path and logging configuration."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.file_path = 'data/processed/csv/'  # Default folder for saving CSV files

    def _save_csv(self, file_path, df, name, index=False):
        """
        Saves the given dataframe to a CSV file.

        Args:
            file_path (str): The directory path to save the CSV.
            df (pd.DataFrame): The dataframe to be saved.
            name (str): The name of the CSV file (without extension).
            index (bool): Whether to include the dataframe index in the CSV file (default is False).
        """
        full_path = get_full_path(file_path)
        try:
            df.to_csv(os.path.join(full_path, f"{name}.csv"), index=index, na_rep='')  # Save CSV with or without index
            logging.info(f"{name.replace('_', ' ').title()} dataframe saved successfully")
        except Exception as e:
            logging.error(f"Error saving dataframe {name}: {e}")

    def save_final_csv(self, w_df=None, a_df=None, df=None, foods_df=None, file_path=None):
        """
        Save final CSV files for workouts, activities, foods, and final data.

        Args:
            w_df (pd.DataFrame, optional): Workouts dataframe.
            a_df (pd.DataFrame, optional): Activities dataframe.
            df (pd.DataFrame, optional): Final merged dataframe.
            foods_df (pd.DataFrame, optional): Foods dataframe.
            file_path (str, optional): Custom file path for saving the CSVs.
        """
        for name, data in zip(['final_df', 'foods_df', 'workouts_df', 'activities_df'], [df, foods_df, w_df, a_df]):
            if data is not None:
                index = name in ['final_df', 'foods_df']  # Include index only for 'final_df' and 'foods_df'
                self._save_csv(self.file_path if file_path is None else file_path, data, name, index)

    def save_tss_values_for_dashboard(self, tss, atl, ctl, tsb, file_path=None):
        """
        Save TSS, ATL, CTL, and TSB metrics for the dashboard.

        Args:
            tss (pd.DataFrame): Training Stress Score (TSS) dataframe.
            atl (pd.DataFrame): Acute Training Load (ATL) dataframe.
            ctl (pd.DataFrame): Chronic Training Load (CTL) dataframe.
            tsb (pd.DataFrame): Training Stress Balance (TSB) dataframe.
            file_path (str, optional): Custom file path for saving the CSVs.
        """
        for name, data in zip(['tss', 'atl', 'ctl', 'tsb'], [tss, atl, ctl, tsb]):
            if data is not None:
                self._save_csv(self.file_path if file_path is None else file_path, data, name, index=True)

    def save_during_process(self, work_df, acti_df, file_path=None):
        """
        Save intermediate workout and activity dataframes during the processing step.

        Args:
            work_df (pd.DataFrame): Workouts dataframe to be saved.
            acti_df (pd.DataFrame): Activities dataframe to be saved.
            file_path (str, optional): Custom file path for saving the CSVs.
        """
        for name, data in zip(['workouts_to_process_df', 'activities_to_process_df'], [work_df, acti_df]):
            if data is not None:
                self._save_csv(self.file_path if file_path is None else file_path, data, name)

    def save_nutrition(self, nutrition_df, file_path=None):
        """
        Save the user's nutrition data.

        Args:
            nutrition_df (pd.DataFrame): The nutrition dataframe to be saved.
            file_path (str, optional): Custom file path for saving the CSV.
        """
        self._save_csv(self.file_path if file_path is None else file_path, nutrition_df, 'user_nutrition')
