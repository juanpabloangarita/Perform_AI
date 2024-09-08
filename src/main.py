# main.py

#from data_processing import process_data, load_csv
from src.data_processing import process_data, load_csv  # Use relative import # WARNING, WHY A DOT BEFORE DATA_PROCESSING

#def main(*uploaded_workouts):
def main(workouts=None):
    # Your app logic here
    print("App is running")
    tss_df, atl_df, ctl_df, tsb_df, w_df, a_df = process_data(workouts)
    
    return tss_df, atl_df, ctl_df, tsb_df, w_df, a_df

if __name__ == "__main__":
    main()
