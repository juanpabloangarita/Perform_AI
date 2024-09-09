# calorie_estimation_models.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn.impute import SimpleImputer

from src.data_processing import *

# Linear Regression model with scaling
def linear_regression_model(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Linear Regression RMSE: {rmse}")
    return pipeline, rmse

# Random Forest model with hyperparameter tuning
def random_forest_model(X_train, X_test, y_train, y_test):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Tuned Random Forest RMSE: {rmse}")
    return best_model, rmse

# Gradient Boosting model with scaling and tuning
def gradient_boosting_model(X_train, X_test, y_train, y_test):
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Tuned Gradient Boosting RMSE: {rmse}")
    return best_model, rmse

# LightGBM model with scaling and tuning
def lightgbm_model(X_train, X_test, y_train, y_test):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    grid_search = GridSearchCV(lgb.LGBMRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Tuned LightGBM RMSE: {rmse}")
    return best_model, rmse

# XGBoost model with scaling and tuning
def xgboost_model(X_train, X_test, y_train, y_test):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Tuned XGBoost RMSE: {rmse}")
    return best_model, rmse

def estimate_calories_without_workout_type(activities_df, past_workouts, future_workouts):
    # Prepare features and labels for the regression models without WorkoutType (for past workouts)
    X_activities_y_hr = activities_df[['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage']].copy()
    y_activities = activities_df['Calories']

    # Prepare features and labels for future workouts without WorkoutType
    X_activities_no_hr = activities_df[['DistanceInMeters', 'TimeTotalInHours']].copy()
    X_activities_no_hr.rename(columns={
        'DistanceInMeters': 'PlannedDistanceInMeters',
        'TimeTotalInHours': 'PlannedDuration'
    }, inplace=True)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_activities_y_hr_imputed = imputer.fit_transform(X_activities_y_hr)
    X_activities_no_hr_imputed = imputer.transform(X_activities_no_hr)

    # Add polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_activities_y_hr_poly = poly.fit_transform(X_activities_y_hr_imputed)
    X_activities_no_hr_poly = poly.transform(X_activities_no_hr_imputed)

    # Split data into training and test sets
    X_train_y_hr, X_test_y_hr, y_train, y_test = train_test_split(X_activities_y_hr_poly, y_activities, test_size=0.2, random_state=42)
    X_train_no_hr, X_test_no_hr, _, _ = train_test_split(X_activities_no_hr_poly, y_activities, test_size=0.2, random_state=42)

    # Train and evaluate the models WITH HEART RATE
    linear_model_y_hr, rmse_lr_y_hr = linear_regression_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    rf_model_y_hr, rmse_rf_y_hr = random_forest_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    gb_model_y_hr, rmse_gb_y_hr = gradient_boosting_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    lgb_model_y_hr, rmse_lgb_y_hr = lightgbm_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    xgb_model_y_hr, rmse_xgb_y_hr = xgboost_model(X_train_y_hr, X_test_y_hr, y_train, y_test)

    # Train and evaluate the models WITHOUT HEART RATE
    linear_model_no_hr, rmse_lr_no_hr = linear_regression_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    rf_model_no_hr, rmse_rf_no_hr = random_forest_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    gb_model_no_hr, rmse_gb_no_hr = gradient_boosting_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    lgb_model_no_hr, rmse_lgb_no_hr = lightgbm_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    xgb_model_no_hr, rmse_xgb_no_hr = xgboost_model(X_train_no_hr, X_test_no_hr, y_train, y_test)

    # Use the best model for past workouts without WorkoutType
    mask_past = past_workouts[['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage']].notna().all(axis=1)
    X_past_workouts = past_workouts.loc[mask_past, ['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage']].copy()
    X_past_workouts_imputed = imputer.transform(X_past_workouts)
    X_past_workouts_poly = poly.transform(X_past_workouts_imputed)
    past_workouts.loc[mask_past, 'EstimatedCalories'] = xgb_model_y_hr.predict(X_past_workouts_poly)

    # Use the best model for future workouts without WorkoutType
    mask_future = future_workouts[['PlannedDistanceInMeters', 'PlannedDuration']].notna().all(axis=1)
    X_future_workouts = future_workouts.loc[mask_future, ['PlannedDistanceInMeters', 'PlannedDuration']].copy()
    X_future_workouts_imputed = imputer.transform(X_future_workouts)
    X_future_workouts_poly = poly.transform(X_future_workouts_imputed)
    future_workouts.loc[mask_future, 'EstimatedCalories'] = rf_model_no_hr.predict(X_future_workouts_poly)

    # Combine past and future workouts back together
    workouts_df = pd.concat([past_workouts, future_workouts])

    return workouts_df, {
        'rmse_lr_y_hr': rmse_lr_y_hr, 'rmse_rf_y_hr': rmse_rf_y_hr, 'rmse_gb_y_hr': rmse_gb_y_hr,
        'rmse_lgb_y_hr': rmse_lgb_y_hr, 'rmse_xgb_y_hr': rmse_xgb_y_hr,
        'rmse_lr_no_hr': rmse_lr_no_hr, 'rmse_rf_no_hr': rmse_rf_no_hr, 'rmse_gb_no_hr': rmse_gb_no_hr,
        'rmse_lgb_no_hr': rmse_lgb_no_hr, 'rmse_xgb_no_hr': rmse_xgb_no_hr
    }


def estimate_calories_without_workout_type_v2(activities_df, past_workouts, future_workouts):
    # Prepare features and labels for the regression models without WorkoutType (for past workouts)
    X_activities_y_hr = activities_df[['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage']].copy()

    y_activities = activities_df['Calories']

    # Prepare features and labels for future workouts without WorkoutType
    X_activities_no_hr = activities_df[['DistanceInMeters', 'TimeTotalInHours']].copy()
    X_activities_no_hr.rename(columns={
        'DistanceInMeters': 'PlannedDistanceInMeters',
        'TimeTotalInHours': 'PlannedDuration'
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
Linear Regression RMSE: 79.33538641583827
Random Forest RMSE: 84.0087350171616
Gradient Boosting RMSE: 75.86140919738551
LIGHTGBM RMSE: 75.37150839370808
XGBOOST RMSE: 79.11720568354129

Without Heart Rate:
Linear Regression RMSE: 98.44755737772297
Random Forest RMSE: 96.13127611614661
Gradient Boosting RMSE: 98.35461364126367
LIGHTGBM RMSE: 90.32195331635167
XGBOOST RMSE: 101.84941904422894


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
def estimate_calories_with_workout_type(activities_df, past_workouts, future_workouts):
    # Prepare features and labels for the regression models with HeartRateAverage and WorkoutType (for past workouts)
    X_activities_y_hr = activities_df[['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage', 'WorkoutType']].copy()

    X_activities_y_hr = pd.get_dummies(X_activities_y_hr)

    y_activities = activities_df['Calories']

    # Prepare features and labels for future workouts with WorkoutType
    X_activities_no_hr = activities_df[['DistanceInMeters', 'TimeTotalInHours', 'WorkoutType']].copy()
    X_activities_no_hr.rename(columns={
        'DistanceInMeters': 'PlannedDistanceInMeters',
        'TimeTotalInHours': 'PlannedDuration'
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
