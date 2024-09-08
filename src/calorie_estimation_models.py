# calorie_estimation_models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

from src.data_processing import *

# Linear Regression model
def linear_regression_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Linear Regression RMSE: {rmse}")
    return model, rmse

# Random Forest model
def random_forest_model(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Random Forest RMSE: {rmse}")
    return model, rmse

# Gradient Boosting (sklearn) model
def gradient_boosting_model(X_train, X_test, y_train, y_test):
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Gradient Boosting RMSE: {rmse}")
    return model, rmse

# LightGBM model
def lightgbm_model(X_train, X_test, y_train, y_test):
    model = lgb.LGBMRegressor(n_estimators=200, max_depth=10, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"LightGBM RMSE: {rmse}")
    return model, rmse


# XGBoost model
def xgboost_model(X_train, X_test, y_train, y_test):
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"XGBoost RMSE: {rmse}")
    return model, rmse


def estimate_calories_with_workout_type(activities_df, past_workouts, future_workouts):
    # Prepare features and labels for the regression models with HeartRateAverage and WorkoutType (for past workouts)
    X_activities_y_hr = activities_df[['Distance', 'Durée', 'Fréquence cardiaque moyenne', 'Type d\'activité']].copy()
    X_activities_y_hr.rename(columns={
        'Distance': 'DistanceInMeters',
        'Durée': 'TimeTotalInHours',
        'Fréquence cardiaque moyenne': 'HeartRateAverage',
        'Type d\'activité': 'WorkoutType'
    }, inplace=True)
    X_activities_y_hr = pd.get_dummies(X_activities_y_hr)
    y_activities = activities_df['Calories']

    # Prepare features and labels for future workouts with WorkoutType
    X_activities_no_hr = activities_df[['Distance', 'Durée', 'Type d\'activité']].copy()
    X_activities_no_hr.rename(columns={
        'Distance': 'PlannedDistanceInMeters',
        'Durée': 'PlannedDuration',
        'Type d\'activité': 'WorkoutType'
    }, inplace=True)
    X_activities_no_hr = pd.get_dummies(X_activities_no_hr)

    # Split data into training and test sets
    X_train_y_hr, X_test_y_hr, y_train, y_test = train_test_split(X_activities_y_hr, y_activities, test_size=0.2, random_state=42)
    X_train_no_hr, X_test_no_hr, _, _ = train_test_split(X_activities_no_hr, y_activities, test_size=0.2, random_state=42)

    # Train and evaluate the models
    linear_model_y_hr, rmse_lr_y_hr = linear_regression_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    rf_model_y_hr, rmse_rf_y_hr = random_forest_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    gb_model_y_hr, rmse_gb_y_hr = gradient_boosting_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    lgb_model_y_hr, rmse_lgb_y_hr = lightgbm_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    xgb_model_y_hr, rmse_xgb_y_hr = xgboost_model(X_train_y_hr, X_test_y_hr, y_train, y_test)

    linear_model_no_hr, rmse_lr_no_hr = linear_regression_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    rf_model_no_hr, rmse_rf_no_hr = random_forest_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    gb_model_no_hr, rmse_gb_no_hr = gradient_boosting_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    lgb_model_no_hr, rmse_lgb_no_hr = lightgbm_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    xgb_model_no_hr, rmse_xgb_no_hr = xgboost_model(X_train_no_hr, X_test_no_hr, y_train, y_test)

    # Use the best model for past workouts with WorkoutType
    mask_past = past_workouts[['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage', 'WorkoutType']].notna().all(axis=1)
    past_workouts = pd.get_dummies(past_workouts, columns=['WorkoutType'])
    workout_type_columns_past = past_workouts.columns[past_workouts.columns.str.startswith('WorkoutType')]
    X_past_workouts = past_workouts.loc[mask_past, ['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage'] + list(workout_type_columns_past)].copy()
    past_workouts.loc[mask_past, 'EstimatedCalories'] = gb_model_y_hr.predict(X_past_workouts)

    # Use the best model for future workouts with WorkoutType
    mask_future = future_workouts[['PlannedDistanceInMeters', 'PlannedDuration', 'WorkoutType']].notna().all(axis=1)
    future_workouts = pd.get_dummies(future_workouts, columns=['WorkoutType'])
    workout_type_columns_future = future_workouts.columns[future_workouts.columns.str.startswith('WorkoutType')]
    X_future_workouts = future_workouts.loc[mask_future, ['PlannedDistanceInMeters', 'PlannedDuration'] + list(workout_type_columns_future)].copy()
    future_workouts.loc[mask_future, 'EstimatedCalories'] = gb_model_no_hr.predict(X_future_workouts)

    # Combine past and future workouts back together
    workouts_df = pd.concat([past_workouts, future_workouts])

    return workouts_df, {
        'rmse_lr_y_hr': rmse_lr_y_hr, 'rmse_rf_y_hr': rmse_rf_y_hr, 'rmse_gb_y_hr': rmse_gb_y_hr,
        'rmse_lgb_y_hr': rmse_lgb_y_hr, 'rmse_xgb_y_hr': rmse_xgb_y_hr,
        'rmse_lr_no_hr': rmse_lr_no_hr, 'rmse_rf_no_hr': rmse_rf_no_hr, 'rmse_gb_no_hr': rmse_gb_no_hr,
        'rmse_lgb_no_hr': rmse_lgb_no_hr, 'rmse_xgb_no_hr': rmse_xgb_no_hr
    }


def estimate_calories_without_workout_type(activities_df, past_workouts, future_workouts):
    # Prepare features and labels for the regression models without WorkoutType (for past workouts)
    X_activities_y_hr = activities_df[['Distance', 'Durée', 'Fréquence cardiaque moyenne']].copy()
    X_activities_y_hr.rename(columns={
        'Distance': 'DistanceInMeters',
        'Durée': 'TimeTotalInHours',
        'Fréquence cardiaque moyenne': 'HeartRateAverage'
    }, inplace=True)

    X_activities_y_hr['HeartRateAverage'] = pd.to_numeric(X_activities_y_hr['HeartRateAverage'], errors='coerce')

    y_activities = activities_df['Calories']

    # Prepare features and labels for future workouts without WorkoutType
    X_activities_no_hr = activities_df[['Distance', 'Durée']].copy()
    X_activities_no_hr.rename(columns={
        'Distance': 'PlannedDistanceInMeters',
        'Durée': 'PlannedDuration'
    }, inplace=True)

    # Split data into training and test sets
    X_train_y_hr, X_test_y_hr, y_train, y_test = train_test_split(X_activities_y_hr, y_activities, test_size=0.2, random_state=42)
    X_train_no_hr, X_test_no_hr, _, _ = train_test_split(X_activities_no_hr, y_activities, test_size=0.2, random_state=42)

    # Train and evaluate the models WITH HEART RATE
    linear_model_y_hr, rmse_lr_y_hr = linear_regression_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    rf_model_y_hr, rmse_rf_y_hr = random_forest_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    gb_model_y_hr, rmse_gb_y_hr = gradient_boosting_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    lgb_model_y_hr, rmse_lgb_y_hr = lightgbm_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    xgb_model_y_hr, rmse_xgb_y_hr = xgboost_model(X_train_y_hr, X_test_y_hr, y_train, y_test)

    # WITHOUT HEART RATE
    linear_model_no_hr, rmse_lr_no_hr = linear_regression_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    rf_model_no_hr, rmse_rf_no_hr = random_forest_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    gb_model_no_hr, rmse_gb_no_hr = gradient_boosting_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    lgb_model_no_hr, rmse_lgb_no_hr = lightgbm_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    xgb_model_no_hr, rmse_xgb_no_hr = xgboost_model(X_train_no_hr, X_test_no_hr, y_train, y_test)

    # Use the best model for past workouts without WorkoutType
    mask_past = past_workouts[['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage']].notna().all(axis=1)
    X_past_workouts = past_workouts.loc[mask_past, ['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage']].copy()
    #past_workouts.loc[mask_past, 'EstimatedCalories'] = gb_model_y_hr.predict(X_past_workouts)
    past_workouts.loc[mask_past, 'EstimatedCalories'] = xgb_model_y_hr.predict(X_past_workouts)

    # Use the best model for future workouts without WorkoutType
    mask_future = future_workouts[['PlannedDistanceInMeters', 'PlannedDuration']].notna().all(axis=1)

    X_future_workouts = future_workouts.loc[mask_future, ['PlannedDistanceInMeters', 'PlannedDuration']].copy()
    future_workouts.loc[mask_future, 'EstimatedCalories'] = rf_model_no_hr.predict(X_future_workouts)

    # Combine past and future workouts back together
    workouts_df = pd.concat([past_workouts, future_workouts])

    return workouts_df, {
        'rmse_lr_y_hr': rmse_lr_y_hr, 'rmse_rf_y_hr': rmse_rf_y_hr, 'rmse_gb_y_hr': rmse_gb_y_hr,
        'rmse_lgb_y_hr': rmse_lgb_y_hr, 'rmse_xgb_y_hr': rmse_xgb_y_hr,
        'rmse_lr_no_hr': rmse_lr_no_hr, 'rmse_rf_no_hr': rmse_rf_no_hr, 'rmse_gb_no_hr': rmse_gb_no_hr,
        'rmse_lgb_no_hr': rmse_lgb_no_hr, 'rmse_xgb_no_hr': rmse_xgb_no_hr
    }


"""BETTER
estimate_calories_without_workout_type

With Heart Rate:
Linear Regression RMSE: 71.39344172120131
Random Forest RMSE: 76.04153312773663
Gradient Boosting RMSE: 65.73461347345477
LIGHTGBM RMSE: 64.21517223921495
XGBOOST RMSE: 76.5706280497429

Without Heart Rate:
Linear Regression RMSE: 86.7448131426331
Random Forest RMSE: 93.76334490225517
Gradient Boosting RMSE: 90.71393979536293
LIGHTGBM RMSE: 84.73938121141626
XGBOOST RMSE: 91.41686969387759
"""


"""
estimate_calories_with_workout_type

With Heart Rate: PAST NO NEED CUZ I HAVE THEM
Linear Regression RMSE: 69.09088489863505
Random Forest RMSE: 74.8078427330651
Gradient Boosting RMSE: 63.804675982059706
LIGHTGBM RMSE: 61.86555013329693
XGBOOST RMSE: 70.25142062749532

Without Heart Rate:
Linear Regression RMSE: 84.73259158153283
Random Forest RMSE: 88.68598038614066
Gradient Boosting RMSE: 87.20099204368888
LIGHTGBM RMSE: 77.05421262471275
XGBOOST RMSE: 82.21859160846236
"""
