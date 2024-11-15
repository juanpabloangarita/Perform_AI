# Perform_AI.src.main.py

from src.data_processing import process_data
from src.data_loader.files_saving import FileSaver

def main(user_data, workouts=None, main_arg=None):
    if main_arg == 'main':
        print("\n\n\nApp is running\n\n\n")
        w_df_calories_estimated_plus_calculated, activities_df, final_df = process_data(user_data, workouts)

        # Save multiple DataFrames (workouts and activities with estimated calories) and the final DataFrame
        FileSaver().save_dfs([w_df_calories_estimated_plus_calculated, activities_df], dfs_names=['workouts_df', 'activities_df'])
        FileSaver().save_dfs(final_df, name='final_df', index=True) # Save final DataFrame

        return "Re-processed Main"
    else:
        return "App already loaded"

if __name__ == "__main__":
    main()
