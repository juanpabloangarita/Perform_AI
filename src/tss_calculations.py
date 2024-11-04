# Perform_AI.src.tss_calculations.py

import numpy as np
import pandas as pd
from params import GIVEN_DATE
from src.data_loader.files_extracting import FileSaver

def calculate_total_tss_and_metrics_from_tss(df, source, given_date=GIVEN_DATE): # NOTE: Everytime i have a new data point, i recalculate the WHOLE metrics again, not efficient
    """
    Calculate total Training Stress Score (TSS) for different workout types and call the calculate_metrics_from_tss to calculate metrics

    Parameters:
        df (pd.DataFrame): DataFrame containing workout data.
        source (str):   Source of data ('data_processing' or other).
                        if 'PlannedDuration' then 'data_processing' else update_final_df from FileLoader
        given_date (str): Date to filter the data.

    Returns:
        pd.DataFrame: Updated DataFrame with total TSS calculated AND TSS DataFrame, ATL, CTL, and TSB DataFrames.
    """
    column_to_use = 'PlannedDuration' if source == 'data_processing' else 'TimeTotalInHours'

    workout_types = {
        'Run': (140, calculating_running_tss),
        'Bike': (136, calculating_cycling_tss),
        'Swim': (30, calculating_swimming_tss)
    }

    # Calculate swimming threshold (s_thrs)
    s_thrs = 100 / (2 + 17 / 60)  # Adjusted as per your previous code

    for workout_type, (avg_hr, calc_function) in workout_types.items():
        mask = df['WorkoutType'] == workout_type

        if workout_type == 'Swim':
            average_pace_swimming = 30  # Average pace for swimming
            calc_function(df, mask, average_pace_swimming, given_date, s_thrs, column_to_use, f"{workout_type}_")
        else:
            calc_function(df, mask, avg_hr, given_date, column_to_use, f"{workout_type}_")

    # Calculate TOTAL TSS
    df['TOTAL TSS'] = df[['Run_TSS Calculated', 'Bike_TSS Calculated', 'Swim_TSS Calculated']].sum(axis=1)

    if source == 'update_final_df':
        df.index = pd.to_datetime(df.index)

    # Calculate ATL, CTL, TSB from TSS
    tss_df, atl_df, ctl_df, tsb_df = calculate_metrics_from_tss(df)
    # This will save the TSS, ATL, CTL, and TSB DataFrames to CSV files
    # The filenames will be 'tss.csv', 'atl.csv', 'ctl.csv', and 'tsb.csv'
    # The index of each DataFrame will be included in the CSV files
    FileSaver().save_dfs([tss_df, atl_df, ctl_df, tsb_df], dfs_names=['tss', 'atl', 'ctl', 'tsb'], index=True)

    return df

def calculating_running_tss(df, mask, avg_hr, date, column, discipline):
    """Calculate TSS for running workouts."""
    df[f'{discipline}TSS Calculated'] = 0.0
    df.loc[mask & (df.index < date), f'{discipline}TSS Calculated'] = (
        df.loc[mask & (df.index < date), 'TimeTotalInHours'] * 60 *
        ((df.loc[mask & (df.index < date), 'HeartRateAverage'] - 42) / (163 - 42)) ** 2
    )
    df.loc[mask & (df.index >= date), f'{discipline}TSS Calculated'] = (
        df.loc[mask & (df.index >= date), column] * 60 *
        ((avg_hr - 42) / (163 - 42)) ** 2
    )

def calculating_cycling_tss(df, mask, avg_hr, date, column, discipline):
    """Calculate TSS for cycling workouts."""
    df[f'{discipline}TSS Calculated'] = 0.0
    df.loc[mask & (df.index < date), f'{discipline}TSS Calculated'] = (
        df.loc[mask & (df.index < date), 'TimeTotalInHours'] * 60 *
        ((df.loc[mask & (df.index < date), 'HeartRateAverage'] - 42) / (157 - 42)) ** 2
    )
    df.loc[mask & (df.index >= date), f'{discipline}TSS Calculated'] = (
        df.loc[mask & (df.index >= date), column] * 60 *
        ((avg_hr - 42) / (157 - 42)) ** 2
    )

    correcting_mask = (df[f'{discipline}TSS Calculated'] == 0) & mask
    df.loc[correcting_mask & (df.index < date), f'{discipline}TSS Calculated'] = (
        df.loc[correcting_mask & (df.index < date), 'TimeTotalInHours'] * 60 *
        ((avg_hr - 42) / (157 - 42)) ** 2
    )

def calculating_swimming_tss(df, mask, avg_pace, date, s_thrs, column, discipline):
    """Calculate TSS for swimming workouts."""
    df[f'{discipline}TSS Calculated'] = 0.0
    pace_per_100m = (df.loc[mask & (df.index < date), 'DistanceInMeters'] / df.loc[mask & (df.index < date), 'TimeTotalInHours']) / 100
    normalized_pace_ratio = pace_per_100m / s_thrs
    normalized_pace_ratio_avg = avg_pace / s_thrs

    df.loc[mask & (df.index < date), f'{discipline}TSS Calculated'] = (
        (normalized_pace_ratio ** 3) * df.loc[mask & (df.index < date), 'TimeTotalInHours'] * 100
    )
    df.loc[mask & (df.index >= date), f'{discipline}TSS Calculated'] = (
        (normalized_pace_ratio_avg ** 3) * df.loc[mask & (df.index >= date), column] * 100
    )

def calculate_metrics_from_tss(df, given_date=GIVEN_DATE):
    """
    Calculate ATL, CTL, and TSB metrics from TSS.

    Parameters:
        df (pd.DataFrame): DataFrame containing TSS data.
        given_date (str): Date for filtering the data.

    Returns:
        tuple: TSS DataFrame, ATL, CTL, and TSB DataFrames.
    """
    group_by = 'Date' if 'Date' in df.columns else df.index
    calc_df = df.groupby(group_by).agg({'TOTAL TSS': 'sum', 'HeartRateAverage': 'mean'})

    # Resampling TSS on a daily basis, summing the values
    tss_df = calc_df[['TOTAL TSS']].resample('D').sum()

    # Constants
    k_atl = 7
    k_ctl = 42

    # Initializing the ATL and CTL DataFrames
    atl_series = pd.Series(index=tss_df.index, data=0.0)
    ctl_series = pd.Series(index=tss_df.index, data=0.0)

    # Smoothing factors
    alpha_atl = 1 - np.exp(-1 / k_atl)
    alpha_ctl = 1 - np.exp(-1 / k_ctl)

    # Initial values
    atl_series.iloc[0] = tss_df.iloc[0] * alpha_atl
    ctl_series.iloc[0] = tss_df.iloc[0] * alpha_ctl

    # Calculate ATL and CTL iteratively
    for i in range(1, len(tss_df)):
        atl_series.iloc[i] = atl_series.iloc[i - 1] * (1 - alpha_atl) + tss_df.iloc[i] * alpha_atl
        ctl_series.iloc[i] = ctl_series.iloc[i - 1] * (1 - alpha_ctl) + tss_df.iloc[i] * alpha_ctl

    tsb_series = ctl_series - atl_series
    tss_df.loc[:given_date] = tss_df.loc[:given_date].replace(0, np.nan)

    return tss_df, atl_series.to_frame(name='ATL'), ctl_series.to_frame(name='CTL'), tsb_series.to_frame(name='TSB')
