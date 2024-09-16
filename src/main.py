# main.py

#from data_processing import process_data, load_csv
from src.data_processing import process_data, save_final_csv, save_tss_values_for_dashboard  # Use relative import # WARNING, WHY A DOT BEFORE DATA_PROCESSING

#def main(*uploaded_workouts):
def main(user_data, workouts=None):
    # Your app logic here
    print("App is running")
    tss_df, atl_df, ctl_df, tsb_df, w_df_calories_estimated_plus_calculated, activities_df, final_df = process_data(user_data, workouts)

    save_tss_values_for_dashboard('data/processed/csv/', tss_df, atl_df, ctl_df, tsb_df) # TODO: here should be sourcer
    save_final_csv('data/processed/csv/', w_df_calories_estimated_plus_calculated, activities_df, final_df) # TODO: here should be sourcer

    return "You got this!"

if __name__ == "__main__":
    main()
