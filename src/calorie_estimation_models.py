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
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
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

# Main function
def estimate_calories(activities_df, past_workouts, future_workouts):
    # Prepare features and labels for the regression models with HeartRateAverage (for past workouts)
    X_activities_y_hr = activities_df[['Distance', 'Durée', 'Fréquence cardiaque moyenne']].copy()
    y_activities = activities_df['Calories']

    # Prepare features and labels without HeartRateAverage (for future workout model)
    X_activities_no_hr = activities_df[['Distance', 'Durée']].copy()

    # Split data into training and test sets for both models
    X_train_y_hr, X_test_y_hr, y_train, y_test = train_test_split(X_activities_y_hr, y_activities, test_size=0.2, random_state=42)
    X_train_no_hr, X_test_no_hr, _, _ = train_test_split(X_activities_no_hr, y_activities, test_size=0.2, random_state=42)

    # Train and evaluate the models for past workouts (with HeartRateAverage)
    linear_model_y_hr, rmse_lr_y_hr = linear_regression_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    rf_model_y_hr, rmse_rf_y_hr = random_forest_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    gb_model_y_hr, rmse_gb_y_hr = gradient_boosting_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    lgb_model_y_hr, rmse_lgb_y_hr = lightgbm_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
    xgb_model_y_hr, rmse_xgb_y_hr = xgboost_model(X_train_y_hr, X_test_y_hr, y_train, y_test)

    # Train and evaluate the models for future workouts (without HeartRateAverage)
    linear_model_no_hr, rmse_lr_no_hr = linear_regression_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    rf_model_no_hr, rmse_rf_no_hr = random_forest_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    gb_model_no_hr, rmse_gb_no_hr = gradient_boosting_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    lgb_model_no_hr, rmse_lgb_no_hr = lightgbm_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    xgb_model_no_hr, rmse_xgb_no_hr = xgboost_model(X_train_no_hr, X_test_no_hr, y_train, y_test)

    # Use the best model for past workouts (with HeartRateAverage)
    X_past_workouts = past_workouts[['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage']].copy().dropna()
    past_workouts['EstimatedCalories'] = gb_model_y_hr.predict(X_past_workouts)

    # Use the best model for future workouts (without HeartRateAverage)
    X_future_workouts = future_workouts[['PlannedDistanceInMeters', 'PlannedDuration']].copy().dropna()
    future_workouts['EstimatedCalories'] = gb_model_no_hr.predict(X_future_workouts)

    # Combine past and future workouts back together
    workouts_df = pd.concat([past_workouts, future_workouts])

    return workouts_df, {
        'rmse_lr_y_hr': rmse_lr_y_hr, 'rmse_rf_y_hr': rmse_rf_y_hr, 'rmse_gb_y_hr': rmse_gb_y_hr,
        'rmse_lgb_y_hr': rmse_lgb_y_hr, 'rmse_xgb_y_hr': rmse_xgb_y_hr,
        'rmse_lr_no_hr': rmse_lr_no_hr, 'rmse_rf_no_hr': rmse_rf_no_hr, 'rmse_gb_no_hr': rmse_gb_no_hr,
        'rmse_lgb_no_hr': rmse_lgb_no_hr, 'rmse_xgb_no_hr': rmse_xgb_no_hr
    }
