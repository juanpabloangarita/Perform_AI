# Perform_AI.src.main.py

from src.data_processing import process_data
from src.data_loader.files_saving import FileSaver

def main(user_data, workouts=None, main_arg=None):
    if main_arg == 'main':
        print("\n\n\nApp is running\n\n\n")
        tss_df, atl_df, ctl_df, tsb_df, w_df_calories_estimated_plus_calculated, activities_df, final_df = process_data(user_data, workouts)

        FileSaver().save_tss_values_for_dashboard(tss_df, atl_df, ctl_df, tsb_df)
        FileSaver().save_raw_and_final_dataframes(
            w_df=w_df_calories_estimated_plus_calculated,
            a_df=activities_df,
            df=final_df
        )
        return "Re-processed Main"
    else:
        return "App already loaded"

if __name__ == "__main__":
    main()
