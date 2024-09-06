import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from src.data_processing import *


# Linear Regression model
def linear_regression_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Linear Regression RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    return model

# Random Forest model
def random_forest_model(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    return model

# Gradient Boosting model
def gradient_boosting_model(X_train, X_test, y_train, y_test):
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Gradient Boosting RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    return model

# Main function
def estimate_calories(activities_df, past_workouts, future_workouts):
    # Prepare features and labels for the regression models
    X_activities = activities_df[['Distance', 'Durée', 'Fréquence cardiaque moyenne']].copy()
    y_activities = activities_df['Calories']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_activities, y_activities, test_size=0.2, random_state=42)

    # Train and evaluate different models
    linear_model = linear_regression_model(X_train, X_test, y_train, y_test)
    rf_model = random_forest_model(X_train, X_test, y_train, y_test)
    gb_model = gradient_boosting_model(X_train, X_test, y_train, y_test)

    # Use the best model (e.g., Gradient Boosting in this case) to estimate calories for past workouts
    X_past_workouts = past_workouts[['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage']].copy()
    X_past_workouts = X_past_workouts.dropna()  # Drop rows with NaN values
    past_workouts['EstimatedCalories'] = gb_model.predict(X_past_workouts)

    # For future workouts, estimate calories using planned duration and distance
    X_future_workouts = future_workouts[['PlannedDistanceInMeters', 'PlannedDuration', 'HeartRateAverage']].copy()
    X_future_workouts = X_future_workouts.dropna()
    future_workouts['EstimatedCalories'] = gb_model.predict(X_future_workouts)

    # Combine past and future workouts back together
    workouts_df = pd.concat([past_workouts, future_workouts])

    return workouts_df
