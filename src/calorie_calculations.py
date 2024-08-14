# calorie_calculations.py

from src.data_processing import *
from src.tss_calculations import *
from params import *


import pandas as pd
import numpy as np

# OBLIGATORY PARAMETERS COME FIRST
# OPTIONAL COME AFTER

def calculate_total_calories(weight=80, height=183, age=41, gender='male', vo2_max=50, resting_hr=42, given_date = GIVEN_DATE, df=None):
#def calculate_total_calories(df, weight=80, height=183, age=41, gender='male', vo2_max=50, resting_hr=42, given_date = GIVEN_DATE):
    # Calculate BMR using Mifflin-St Jeor Equation
    if gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    # Derive the K constant from VO2 max, MHR, and Resting HR
    def calculate_k_constant(vo2_max, max_hr, resting_hr):
        return vo2_max / (max_hr - resting_hr)

    # Calculate MET based on the heart rate, resting HR, and K constant
    def calculate_met(avg_hr_during_effort, resting_hr, k_constant):
        return (avg_hr_during_effort * k_constant) / resting_hr

    # Calorie calculation function from MET
    def calculate_calories(met, weight, duration_hrs):
        return met * weight * duration_hrs

    # Estimate Maximum Heart Rate
    max_hr = 220 - age

    # Calculate the K constant based on VO2 max, Maximum Heart Rate, and Resting Heart Rate
    k_constant = calculate_k_constant(vo2_max, max_hr, resting_hr)

    # Calculate calories burned for each activity
    def calculate_calories_from_tss(tss, avg_hr_during_effort, resting_hr, weight, k_constant):
        # Calculate METs
        met = calculate_met(avg_hr_during_effort, resting_hr, k_constant)

        # Duration is derived from TSS; this is a rough estimate
        duration_hrs = tss / 100  # Roughly assumes TSS of 100 corresponds to 1 hour of activity at threshold

        # Calculate Calories Burned
        return calculate_calories(met, weight, duration_hrs)

    # Apply the calculation for running, cycling, and swimming
    df['RunningCalories'] = df.apply(lambda row: calculate_calories_from_tss(row['rTSS Calculated'], row['HeartRateAverage'], resting_hr, weight, k_constant), axis=1)
    df['CyclingCalories'] = df.apply(lambda row: calculate_calories_from_tss(row['cTSS Calculated'], row['HeartRateAverage'], resting_hr, weight, k_constant), axis=1)
    df['SwimmingCalories'] = df.apply(lambda row: calculate_calories_from_tss(row['sTSS Calculated'], row['HeartRateAverage'], resting_hr, weight, k_constant), axis=1)

    # Sum up the active calories
    df['TotalActiveCalories'] = df['RunningCalories'] + df['CyclingCalories'] + df['SwimmingCalories']

    df['TotalPassiveCalories'] = bmr

    # Calculate Total Daily Energy Expenditure
    df['TDEE'] = df['TotalPassiveCalories'] + df['TotalActiveCalories']

    return df
