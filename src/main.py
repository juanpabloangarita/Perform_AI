# Perform_AI.src.main.py

from src.data_processing import process_data
from src.data_loader.files_saving import FileSaver

def main(user_data, workouts=None, main_arg=None):
    if main_arg == 'main':
        print("\n\n\nApp is running\n\n\n")
        tss_df, atl_df, ctl_df, tsb_df, w_df_calories_estimated_plus_calculated, activities_df, final_df = process_data(user_data, workouts)
        # This will save the TSS, ATL, CTL, and TSB DataFrames to CSV files
        # The filenames will be 'tss.csv', 'atl.csv', 'ctl.csv', and 'tsb.csv'
        # The index of each DataFrame will be included in the CSV files
        FileSaver().save_dfs([tss_df, atl_df, ctl_df, tsb_df], dfs_names=['tss', 'atl', 'ctl', 'tsb'], index=True)

        # Save multiple DataFrames (workouts and activities with estimated calories) and the final DataFrame
        FileSaver().save_dfs([w_df_calories_estimated_plus_calculated, activities_df], dfs_names=['workouts_df', 'activities_df'])
        FileSaver().save_dfs(final_df, name='final_df', index=True) # Save final DataFrame

        return "Re-processed Main"
    else:
        return "App already loaded"

if __name__ == "__main__":
    main()
