# Perform_AI.src.data_processing.py

from src.data_helpers import (
    process_date_column,
    print_metrics_or_data,
    create_models_and_predict,
    create_nixtla_and_predict
)
from src.calorie_calculations import calculate_total_calories
import pandas as pd
from src.data_loader.files_extracting import FileLoader
from src.data_processor import DataProcessor


def process_data(user_data, workouts=None):
    """Process and prepare data, estimate calories, and combine results."""
    sourcer = FileLoader()
    # NOTE: I am loading the csv files that i have by default, corresponding to my info, how should be the behaviour when running the app for someone that has not put their own info?
    sourcer.load_initial_csv_files()
    tmp_workouts, activities_df = sourcer.load_dfs(name_s=['workouts_df', 'activities_df'], file_path='data/raw/csv')
    workouts_df = workouts if workouts is not None else tmp_workouts

    data_processor = DataProcessor(workouts_df, activities_df)
    workouts_df = data_processor.workouts_df
    activities_df = data_processor.activities_df

    w_df_calories_calculated = calculate_total_calories(user_data, df=workouts_df)
    print_metrics_or_data('both', w_df_tmp=workouts_df, act_df_tmp=activities_df)

    # Model creation and predictions
    X_activities = activities_df.rename(columns={'TimeTotalInHours': 'TotalDuration'})
    y_activities = activities_df['Calories']
    w_df_calories_estimated, rmse_results = create_models_and_predict(X_activities, y_activities, workouts_df)
    print_metrics_or_data('rmse', rmse=rmse_results)
    print_metrics_or_data('both', w_df_tmp=w_df_calories_estimated, act_df_tmp=activities_df)

    # Nixtla forecast
    forecast_result = create_nixtla_and_predict(X_activities, y_activities, workouts_df)

    # Combine estimated and calculated calories
    w_df_calories_estimated.reset_index(inplace=True)
    w_df_calories_estimated['Date'] = pd.to_datetime(w_df_calories_estimated['Date'], format='%Y-%m-%d')
    w_df_calories_estimated = pd.merge(w_df_calories_estimated, forecast_result, on='Date', how='left')
    w_df_calories_estimated.set_index('Date', inplace=True)

    final_columns = ['WorkoutType', 'Title', 'WorkoutDescription', 'CoachComments', 'HeartRateAverage',
                     'TimeTotalInHours', 'DistanceInMeters', 'Run_Cal', 'Bike_Cal', 'Swim_Cal',
                     'TotalPassiveCal', 'CalculatedActiveCal', 'EstimatedActiveCal', 'AutoARIMA',
                     'AutoARIMA-lo-95', 'AutoARIMA-hi-95', 'Calories', 'CaloriesSpent', 'CaloriesConsumed','Run_TSS Calculated', 'Bike_TSS Calculated', 'Swim_TSS Calculated', 'TOTAL TSS']

    w_df_calories_estimated_reset = process_date_column(w_df_calories_estimated, standardize=True)
    w_df_calories_calculated_reset = process_date_column(w_df_calories_calculated, standardize=True)
    activities_df_reset = process_date_column(activities_df, standardize=True)

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
    final_df['Real TSS'] = 0.0


    return w_df_combined, activities_df, final_df
