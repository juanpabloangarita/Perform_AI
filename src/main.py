# main.py
import os
import pandas as pd

from src.data_processing import process_data, save_final_csv, save_tss_values_for_dashboard, get_full_path  # Use relative import # WARNING, WHY A DOT BEFORE DATA_PROCESSING

#def main(*uploaded_workouts):
def main(user_data, workouts=None, main_arg=None):

    full_path = get_full_path('data/processed/csv/')
    final_csv_path = os.path.join(full_path, "final_df.csv") # NOTE: I AM NOT USING THIS

    if os.path.exists(final_csv_path) and main_arg != 'main': # TODO: i still need to check as well for atl ctl tsb, tss, activities_df and workouts
        print("File already exists. Loading existing final_df.csv")
        return "You got this!"
    else:
        print("App is running")
        tss_df, atl_df, ctl_df, tsb_df, w_df_calories_estimated_plus_calculated, activities_df, final_df = process_data(user_data, workouts)

        save_tss_values_for_dashboard('data/processed/csv/', tss_df, atl_df, ctl_df, tsb_df) # TODO: here should be sourcer
        save_final_csv(
            'data/processed/csv/',
            w_df=w_df_calories_estimated_plus_calculated,
            a_df=activities_df,
            df=final_df
        ) # TODO: here should be sourcer
        return "Re-processed Main"

if __name__ == "__main__":
    main()
