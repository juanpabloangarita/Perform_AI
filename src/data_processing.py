# Perform_AI.src.data_processing.py

from src.data_helpers import (
    process_date_column,
    print_metrics_or_data,
    create_models_and_predict,
    create_nixtla_and_predict
)

from src.tss_calculations import calculate_total_tss_and_metrics_from_tss
from src.calorie_calculations import calculate_total_calories
import pandas as pd
from src.data_loader.files_extracting import FileLoader
from src.data_processor import DataProcessor
from params import CLOUD_ON

def process_data(user_data, workouts=None):
    """Process and prepare data, estimate calories, and combine results."""
    sourcer = FileLoader()
    if CLOUD_ON == 'no':
        sourcer.load_initial_csv_files()

    tmp_workouts, activities_df = sourcer.load_dfs(name_s=['workouts_df', 'activities_df'], file_path='data/raw/csv')
    workouts_df = workouts if workouts is not None else tmp_workouts

    data_processor = DataProcessor(workouts_df, activities_df)
    workouts_df = data_processor.workouts_df
    activities_df = data_processor.activities_df


    w_df, tss_df, atl_df, ctl_df, tsb_df = calculate_total_tss_and_metrics_from_tss(workouts_df, 'data_processing')
    w_df_calories_calculated = calculate_total_calories(user_data, df=w_df)
    print_metrics_or_data('both', w_df_tmp=w_df, act_df_tmp=activities_df)

    # Model creation and predictions
    X_activities = activities_df.rename(columns={'TimeTotalInHours': 'TotalDuration'})
    y_activities = activities_df['Calories']
    w_df_calories_estimated, rmse_results = create_models_and_predict(X_activities, y_activities, w_df)
    print_metrics_or_data('rmse', rmse=rmse_results)
    print_metrics_or_data('both', w_df_tmp=w_df_calories_estimated, act_df_tmp=activities_df)

    # Nixtla forecast
    forecast_result = create_nixtla_and_predict(X_activities, y_activities, w_df)

    # Combine estimated and calculated calories
    w_df_calories_estimated.reset_index(inplace=True)
    w_df_calories_estimated['Date'] = pd.to_datetime(w_df_calories_estimated['Date'], format='%Y-%m-%d')
    w_df_calories_estimated = pd.merge(w_df_calories_estimated, forecast_result, on='Date', how='left')
    w_df_calories_estimated.set_index('Date', inplace=True)

    final_columns = ['WorkoutType', 'Title', 'WorkoutDescription', 'CoachComments', 'HeartRateAverage',
                     'TimeTotalInHours', 'DistanceInMeters', 'Run_Cal', 'Bike_Cal', 'Swim_Cal',
                     'TotalPassiveCal', 'CalculatedActiveCal', 'EstimatedActiveCal', 'AutoARIMA',
                     'AutoARIMA-lo-95', 'AutoARIMA-hi-95', 'Calories', 'CaloriesSpent', 'CaloriesConsumed']

    w_df_calories_estimated = process_date_column(w_df_calories_estimated, standardize=True)
    w_df_calories_calculated = process_date_column(w_df_calories_calculated, standardize=True)
    activities_df = process_date_column(activities_df, standardize=True)

    # Reset the index to ensure the 'date' is a column and not part of the index
    w_df_calories_estimated_reset = w_df_calories_estimated.reset_index()
    w_df_calories_calculated_reset = w_df_calories_calculated.reset_index()
    activities_df_reset = activities_df.reset_index()

    # Concatenate calculated and estimated calories
    w_df_combined = pd.concat([w_df_calories_estimated_reset, w_df_calories_calculated_reset], axis=1, join='inner')

    final_df = pd.concat([w_df_combined, activities_df_reset[['Date', 'Calories']]], axis=1)

    # Drop duplicate 'Date' columns if they exist
    final_df = final_df.loc[:,~final_df.columns.duplicated()]
    final_df = final_df.set_index('Date')
    final_df = final_df.reindex(columns=final_columns, fill_value=0.0)
    numeric_cols = final_df.select_dtypes(include=['float64', 'int64']).columns
    final_df[numeric_cols] = final_df[numeric_cols].fillna(0.0)
    final_df['ComplianceStatus'] = ''
    final_df['TSS'] = 0.0 # NOTE: probably this could be inserted as final columns, and used the reindex fill_value = 0.0 or a dictionary for this and before line


    return tss_df, atl_df, ctl_df, tsb_df, w_df_combined, activities_df, final_df
