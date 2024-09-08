# tss_calculations.py

import numpy as np
import pandas as pd

from src.data_processing import *
from params import *


def calculate_total_tss(df, given_date = GIVEN_DATE):
    # calculate running tss
    r_mask = df['WorkoutType']=='Run'
    average_hr_running = 140
    calculating_running_tss(df, r_mask, average_hr_running, given_date)

    # calculate cycling tss
    c_mask = df['WorkoutType']=='Bike'
    average_hr_cycling = 136
    calculating_cycling_tss(df, c_mask, average_hr_cycling, given_date)

    # calculate swimming tss
    s_mask = df['WorkoutType']=='Swim'
    # swimming threshold is meters per minutes
    s_thrs = 100/(2 + 17/60)
    average_pace_swimming = 30
    calculating_swimming_tss(df, s_mask, average_pace_swimming, given_date, s_thrs)

    # Calculate TOTAL TSS
    df['TOTAL TSS'] = df['rTSS Calculated']+ df['cTSS Calculated']+ df['sTSS Calculated']

    return df


def calculating_running_tss(df, mask, average, date, discipline='r'):
    # calculating running TSS
    df[f'{discipline}TSS Calculated'] = float(0)
    df.loc[mask & (df.index < date),f'{discipline}TSS Calculated'] = df.loc[mask & (df.index < date),'TimeTotalInHours']*60 * ((df.loc[mask & (df.index < date),'HeartRateAverage']- 42)/(163-42))**2
    df.loc[mask & (df.index >= date),f'{discipline}TSS Calculated'] = df.loc[mask & (df.index >= date),'PlannedDuration']*60 * ((average- 42)/(163-42))**2
    df.fillna({f'{discipline}TSS Calculated': float(0)}, inplace=True)
    #return df


def calculating_cycling_tss(df, mask, average, date, discipline='c'):
    # calculating cycling TSS
    df[f'{discipline}TSS Calculated'] = float(0)
    df.loc[mask & (df.index < date),f'{discipline}TSS Calculated'] = df.loc[mask & (df.index < date),'TimeTotalInHours']*60 * ((df.loc[mask & (df.index < date),'HeartRateAverage']- 42)/(157-42))**2
    df.loc[mask & (df.index >= date),f'{discipline}TSS Calculated'] = df.loc[mask & (df.index >= date),'PlannedDuration']*60 * ((average- 42)/(157-42))**2
    df.fillna({f'{discipline}TSS Calculated': float(0)}, inplace=True)

    # after this, there will be cTSS 0 values on bike, of heart rate that wasn't detected, correction here
    correcting_mask = (df[f'{discipline}TSS Calculated']==0) & mask
    df.loc[correcting_mask & (df.index < date),f'{discipline}TSS Calculated'] = df.loc[correcting_mask & (df.index < date),'TimeTotalInHours']*60 * ((average- 42)/(157-42))**2
    #return df


def calculating_swimming_tss(df, mask, average, date, s_thrs ,discipline='s'):
    # calculating swimming TSS
    df[f'{discipline}TSS Calculated'] = float(0)

    # Calculate the pace in meters per hour and convert to pace per 100 meters
    pace_per_100m = (df.loc[mask & (df.index < date),'DistanceInMeters'] / df.loc[mask & (df.index < date), 'TimeTotalInHours']) / 100

    # Calculate the normalized pace ratio
    normalized_pace_ratio = pace_per_100m / s_thrs

    # Calculate the normalized pace ratio averaged
    normalized_pace_ratio_averaged = average / s_thrs

    # Calculate the TSS
    df.loc[mask & (df.index < date),f'{discipline}TSS Calculated'] = (normalized_pace_ratio ** 3) * df.loc[mask & (df.index < date), 'TimeTotalInHours'] * 100
    df.loc[mask & (df.index >= date),f'{discipline}TSS Calculated'] = (normalized_pace_ratio_averaged ** 3) * df.loc[mask & (df.index >= date), 'PlannedDuration'] * 100


    df.fillna({f'{discipline}TSS Calculated': float(0)}, inplace=True)
    #return df


def calculate_metrics_from_tss(df, given_date = GIVEN_DATE):
    calc_df = df.groupby('Date').agg({'TOTAL TSS': 'sum', 'HeartRateAverage': 'mean'})

    # Resampling TSS on a daily basis, summing the values
    tss_df = calc_df[['TOTAL TSS']].resample('D').sum()

    # Constants
    k_atl = 7
    k_ctl = 42

    # Initializing the ATL and CTL DataFrames with the same index as tss_df
    atl_series = pd.Series(index=tss_df.index, data=0.0)
    ctl_series = pd.Series(index=tss_df.index, data=0.0)

    # Smoothing factors
    alpha_atl = 1 - np.exp(-1/k_atl)
    alpha_ctl = 1 - np.exp(-1/k_ctl)
    # Initial values (assuming starting values of 0 for simplicity)
    atl_series.iloc[0] = tss_df.iloc[0] * alpha_atl
    ctl_series.iloc[0] = tss_df.iloc[0] * alpha_ctl

    # Calculate ATL and CTL iteratively
    for i in range(1, len(tss_df)):
        atl_series.iloc[i] = atl_series.iloc[i-1] * (1 - alpha_atl) + tss_df.iloc[i] * alpha_atl
        ctl_series.iloc[i] = ctl_series.iloc[i-1] * (1 - alpha_ctl) + tss_df.iloc[i] * alpha_ctl

    # TSB is the difference between CTL and ATL
    tsb_series = ctl_series - atl_series
    tss_df.loc[:given_date] = tss_df.loc[:given_date].replace(0, np.nan) #WARNING DOES THIS EQUAL TO <given_date as it should be?

    # Convert Series to DataFrames with corresponding columns
    atl_df = atl_series.to_frame(name='ATL')
    ctl_df = ctl_series.to_frame(name='CTL')
    tsb_df = tsb_series.to_frame(name='TSB')
    return tss_df, atl_df, ctl_df, tsb_df
