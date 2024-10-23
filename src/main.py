# main.py
import os
import pandas as pd

from src.data_processing import process_data, save_tss_values_for_dashboard
from src.data_loader.files_saving import Sourcer
from src.data_loader.get_full_path import get_full_path

#def main(*uploaded_workouts):
def main(user_data, workouts=None, main_arg=None):

    full_path = get_full_path('data/processed/csv/') # TODO: LOAD
    final_csv_path = os.path.join(full_path, "final_df.csv")

    if os.path.exists(final_csv_path) and main_arg != 'main': # TODO: i still need to check as well for atl ctl tsb, tss, activities_df and workouts
        print("File already exists. Loading existing final_df.csv")
        return "You got this!"
    else:
        print("App is running")
        tss_df, atl_df, ctl_df, tsb_df, w_df_calories_estimated_plus_calculated, activities_df, final_df = process_data(user_data, workouts)

        Sourcer().save_tss_values_for_dashboard(tss_df, atl_df, ctl_df, tsb_df)

        Sourcer().save_final_csv(
            w_df=w_df_calories_estimated_plus_calculated,
            a_df=activities_df,
            df=final_df
        )
        return "Re-processed Main"

if __name__ == "__main__":
    main()
