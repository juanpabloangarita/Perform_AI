# calorie_calculations.py

from src.data_processing import *
from src.tss_calculations import *
from params import *


import pandas as pd
import numpy as np

# OBLIGATORY PARAMETERS COME FIRST
# OPTIONAL COME AFTER

def calculate_total_calories(weight=None, height=None, age=None, gender=None, vo2_max=None, resting_hr=None, from_where=None, df=None):
    # Baseline Metabolic Rate (BMR)
    # Calculate BMR using Mifflin-St Jeor Equation
    if gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    # Estimate NEAT (Non-Exercise Activity Thermogenesis)
    # Sedentary: NEAT is about 10% of BMR.
    # Lightly active: NEAT is about 15-20% of BMR.
    # if activity_level == 'sedentary':
    #     neat = 0.10 * bmr
    # elif activity_level == 'lightly active':
    #     neat = 0.15 * bmr  # Use 15% as the lower estimate for lightly active
    # else:
    #     neat = 0.20 * bmr  # Use 20% for a higher estimate of light activity

    neat = 0.216 * bmr # Mine given Garmin
    # Total passive calories (BMR + NEAT)
    passive_calories = bmr + neat

    if from_where == 'streamlit':
        return passive_calories, bmr
    else:
        # Derive the K constant from VO2 max, MHR, and Resting HR
        def calculate_k_constant(vo2_max, max_hr, resting_hr):
            return vo2_max / (max_hr - resting_hr)

        # Metabolic Equivalent of Task
        # Calculate MET based on the heart rate, resting HR, and K constant
        def calculate_met(avg_hr_during_effort, resting_hr, k_constant):
            return (avg_hr_during_effort * k_constant) / resting_hr

        # Metabolic Equivalent of Task
        # Calculate MET based on the heart rate, resting HR, and K constant
        def calculate_met_2(avg_hr_during_effort, k_constant):
            return ((avg_hr_during_effort * duration )/weight)* k_constant

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

        df['TotalPassiveCalories'] = passive_calories
        # Calculate Total Daily Energy Expenditure
        df['TotalCalories'] = df['TotalPassiveCalories'] + df['TotalActiveCalories']

        return df[['RunningCalories','CyclingCalories', 'SwimmingCalories' ,'TotalPassiveCalories', 'TotalActiveCalories', 'TotalCalories']]
