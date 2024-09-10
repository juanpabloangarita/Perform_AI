# calorie_estimation_models.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from src.data_processing import *
import joblib

file_path = 'data/processed/models/'
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located - src
dir_script_dir = os.path.dirname(script_dir) # Get the directory where the previous dir is located - PerformAI
full_path = os.path.join(dir_script_dir, file_path)  # Construct the full path


# Helper function to save models
def save_model(model, model_name):
    model_file = os.path.join(full_path, f"{model_name}.pkl")
    joblib.dump(model, model_file)
    print(f"Model {model_name} saved successfully.")
    print(f"Model {model_name} saved successfully at {model_file}.")

# Helper function to load models
def load_model(model_name):
    try:
        model_file = os.path.join(full_path, f"{model_name}.pkl")
        model = joblib.load(model_file)
        print(f"Model {model_name} loaded successfully from {model_file}.")
        return model
    except FileNotFoundError:
        print(f"Model {model_name} not found. Training a new one.")
        return None


# Modify each model function to check for saved models
def linear_regression_model(model_name, X_train, X_test, y_train, y_test):
    model = load_model(model_name)  # Try loading the saved model

    if model is None:  # If the model is not saved, train and save it
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
        pipeline.fit(X_train, y_train)
        save_model(pipeline, model_name)  # Save the trained model
    else:
        pipeline = model  # Use the loaded model

    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Linear Regression RMSE: {rmse}")
    return pipeline, rmse


# Modify the Random Forest function similarly
def random_forest_model(model_name, X_train, X_test, y_train, y_test):
    model = load_model(model_name)  # Try loading the saved model

    if model is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        save_model(best_model, model_name)  # Save the model
    else:
        best_model = model

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Tuned Random Forest RMSE: {rmse}")
    return best_model, rmse


# Apply the same logic for Gradient Boosting
def gradient_boosting_model(model_name, X_train, X_test, y_train, y_test):
    model = load_model(model_name)  # Try loading the saved model

    if model is None:
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        save_model(best_model, model_name)
    else:
        best_model = model

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Tuned Gradient Boosting RMSE: {rmse}")
    return best_model, rmse


# Apply the same logic for LightGBM
def lightgbm_model(model_name, X_train, X_test, y_train, y_test):
    model = load_model(model_name)  # Try loading the saved model

    if model is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'learning_rate': [0.01, 0.05, 0.1]
        }
        grid_search = GridSearchCV(lgb.LGBMRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        save_model(best_model, model_name)
    else:
        best_model = model

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Tuned LightGBM RMSE: {rmse}")
    return best_model, rmse


# Apply the same logic for XGBoost
def xgboost_model(model_name, X_train, X_test, y_train, y_test):
    model = load_model(model_name)  # Try loading the saved model

    if model is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        grid_search = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        save_model(best_model, model_name)
    else:
        best_model = model

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Tuned XGBoost RMSE: {rmse}")
    return best_model, rmse


# Define a function to handle training and evaluation
def train_and_evaluate(model_name, X_train, X_test, y_train, y_test, model_func):
    model, rmse = model_func(model_name, X_train, X_test, y_train, y_test)
    print(f"{model_name} RMSE: {rmse}")
    return model


def prepare_features_for_calorie_estimation(activities_df, past_workouts, future_workouts):
    ## ACTIVITIES
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
    # Impute missing values for features with HeartRateAverage
    imputer_y_hr = KNNImputer()
    X_activities_y_hr_imputed = imputer_y_hr.fit_transform(X_activities_y_hr)

    # Impute missing values for features without HeartRateAverage
    imputer_no_hr = KNNImputer()
    X_activities_no_hr_imputed = imputer_no_hr.fit_transform(X_activities_no_hr)

    # Add polynomial features
    poly_y_hr = PolynomialFeatures(degree=2, include_bias=False)
    X_activities_y_hr_poly = poly_y_hr.fit_transform(X_activities_y_hr_imputed)
    poly_no_hr = PolynomialFeatures(degree=2, include_bias=False)
    X_activities_no_hr_poly = poly_no_hr.fit_transform(X_activities_no_hr_imputed)

    ## PAST WORKOUTS
    mask_past = past_workouts[['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage']].notna().all(axis=1)
    X_past_workouts = past_workouts.loc[mask_past, ['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage']].copy()
    X_past_workouts_imputed = imputer_y_hr.transform(X_past_workouts)
    X_past_workouts_poly = poly_y_hr.transform(X_past_workouts_imputed)

    ## FUTURE WORKOUTS
    mask_future = future_workouts[['PlannedDistanceInMeters', 'PlannedDuration']].notna().all(axis=1)
    X_future_workouts = future_workouts.loc[mask_future, ['PlannedDistanceInMeters', 'PlannedDuration']].copy()
    X_future_workouts_imputed = imputer_no_hr.transform(X_future_workouts)
    X_future_workouts_poly = poly_no_hr.transform(X_future_workouts_imputed)

    return X_activities_y_hr_poly, X_activities_no_hr_poly, y_activities, X_past_workouts_poly, mask_past, X_future_workouts_poly, mask_future


def estimate_calories(activities_df, past_workouts, future_workouts):
    (
        X_activities_y_hr_poly,
        X_activities_no_hr_poly,
        y_activities,
        X_past_workouts_poly,
        mask_past,
        X_future_workouts_poly,
        mask_future
    ) = prepare_features_for_calorie_estimation(
        activities_df, past_workouts, future_workouts
    )
    # Split data into training and test sets
    X_train_y_hr, X_test_y_hr, y_train, y_test = train_test_split(X_activities_y_hr_poly, y_activities, test_size=0.2, random_state=42)
    X_train_no_hr, X_test_no_hr, _, _ = train_test_split(X_activities_no_hr_poly, y_activities, test_size=0.2, random_state=42)

    def create_model_configs(models, X_train_hr, X_test_hr, X_train_no_hr, X_test_no_hr):
        configs = []
        # First add configurations with HR
        for model_name, model_func in models:
            configs.append({
                "name": f"{model_name} with HR",
                "X_train": X_train_hr,
                "X_test": X_test_hr,
                "model_func": model_func
            })

        # Then add configurations without HR
        for model_name, model_func in models:
            configs.append({
                "name": f"{model_name} without HR",
                "X_train": X_train_no_hr,
                "X_test": X_test_no_hr,
                "model_func": model_func
            })

        return configs

    # Define model names and corresponding functions
    models = [
        ("Linear Regression", linear_regression_model),
        ("Random Forest", random_forest_model),
        ("Gradient Boosting", gradient_boosting_model),
        ("LightGBM", lightgbm_model),
        ("XGBoost", xgboost_model)
    ]

    # Create model configurations
    model_configs = create_model_configs(models, X_train_y_hr, X_test_y_hr, X_train_no_hr, X_test_no_hr)

    # Iterate through each model configuration and train
    for config in model_configs:
        model = train_and_evaluate(config["name"], config["X_train"], config["X_test"], y_train, y_test, config["model_func"])
        config['model'] = model

    xgb_model_y_hr = model_configs[4]['model']
    linear_model_no_hr = model_configs[5]['model']

    # Use the best model for past workouts without WorkoutType
    past_workouts.loc[mask_past, 'EstimatedCalories'] = xgb_model_y_hr.predict(X_past_workouts_poly)

    # Use the best model for future workouts without WorkoutType
    future_workouts.loc[mask_future, 'EstimatedCalories'] =  linear_model_no_hr.predict(X_future_workouts_poly)

    # Combine past and future workouts back together
    workouts_df = pd.concat([past_workouts, future_workouts])

    return workouts_df


###################




def estimate_calories_with_workout_type(activities_df, past_workouts, future_workouts, best):
    # BEST PERFORMANCE
    # Add 'WorkoutType' as a categorical feature (using one-hot encoding)
    one_hot_encoder = OneHotEncoder()

    # One-hot encode WorkoutType in activities_df
    activities_encoded = pd.DataFrame(one_hot_encoder.fit_transform(activities_df[['WorkoutType']]).toarray(),
                                      columns=one_hot_encoder.get_feature_names_out(['WorkoutType']),
                                      index=activities_df.index)
    activities_df = pd.concat([activities_df, activities_encoded], axis=1)

    # Prepare features and labels for the regression models WITH WorkoutType (for past workouts)
    X_activities_y_hr = activities_df[['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage'] + list(activities_encoded.columns)].copy()
    y_activities = activities_df['Calories']

    # Prepare features and labels for future workouts WITH WorkoutType
    X_activities_no_hr = activities_df[['DistanceInMeters', 'TimeTotalInHours'] + list(activities_encoded.columns)].copy()
    X_activities_no_hr.rename(columns={
        'DistanceInMeters': 'PlannedDistanceInMeters',
        'TimeTotalInHours': 'PlannedDuration'
    }, inplace=True)

    # Handle missing values
    # Impute missing values for features with HeartRateAverage
    imputer_y_hr = KNNImputer()
    X_activities_y_hr_imputed = imputer_y_hr.fit_transform(X_activities_y_hr)

    # Impute missing values for features without HeartRateAverage
    imputer_no_hr = KNNImputer()
    X_activities_no_hr_imputed = imputer_no_hr.fit_transform(X_activities_no_hr)

    # Add polynomial features
    poly_y_hr = PolynomialFeatures(degree=2, include_bias=False)
    X_activities_y_hr_poly = poly_y_hr.fit_transform(X_activities_y_hr_imputed)
    poly_no_hr = PolynomialFeatures(degree=2, include_bias=False)
    X_activities_no_hr_poly = poly_no_hr.fit_transform(X_activities_no_hr_imputed)

    # Split data into training and test sets
    X_train_y_hr, X_test_y_hr, y_train, y_test = train_test_split(X_activities_y_hr_poly, y_activities, test_size=0.2, random_state=42)
    X_train_no_hr, X_test_no_hr, _, _ = train_test_split(X_activities_no_hr_poly, y_activities, test_size=0.2, random_state=42)

    if best:
        # Train and evaluate the models WITH WorkoutType and HeartRateAverage
        # linear_model_y_hr, rmse_lr_y_hr = linear_regression_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
        # rf_model_y_hr, rmse_rf_y_hr = random_forest_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
        gb_model_y_hr, rmse_gb_y_hr = gradient_boosting_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
        # lgb_model_y_hr, rmse_lgb_y_hr = lightgbm_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
        # xgb_model_y_hr, rmse_xgb_y_hr = xgboost_model(X_train_y_hr, X_test_y_hr, y_train, y_test)

        # Train and evaluate the models WITH WorkoutType and without HeartRateAverage
        # linear_model_no_hr, rmse_lr_no_hr = linear_regression_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
        # rf_model_no_hr, rmse_rf_no_hr = random_forest_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
        # gb_model_no_hr, rmse_gb_no_hr = gradient_boosting_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
        # lgb_model_no_hr, rmse_lgb_no_hr = lightgbm_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
        xgb_model_no_hr, rmse_xgb_no_hr = xgboost_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    else:
        # Train and evaluate the models WITH WorkoutType and HeartRateAverage
        linear_model_y_hr, rmse_lr_y_hr = linear_regression_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
        rf_model_y_hr, rmse_rf_y_hr = random_forest_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
        gb_model_y_hr, rmse_gb_y_hr = gradient_boosting_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
        lgb_model_y_hr, rmse_lgb_y_hr = lightgbm_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
        xgb_model_y_hr, rmse_xgb_y_hr = xgboost_model(X_train_y_hr, X_test_y_hr, y_train, y_test)

        # Train and evaluate the models WITH WorkoutType and without HeartRateAverage
        linear_model_no_hr, rmse_lr_no_hr = linear_regression_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
        rf_model_no_hr, rmse_rf_no_hr = random_forest_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
        gb_model_no_hr, rmse_gb_no_hr = gradient_boosting_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
        lgb_model_no_hr, rmse_lgb_no_hr = lightgbm_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
        xgb_model_no_hr, rmse_xgb_no_hr = xgboost_model(X_train_no_hr, X_test_no_hr, y_train, y_test)

    # Use the best model for past workouts with WorkoutType
    mask_past = past_workouts[['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage']].notna().all(axis=1)
    past_workouts_encoded = pd.DataFrame(one_hot_encoder.transform(past_workouts[['WorkoutType']]).toarray(),
                                         columns=one_hot_encoder.get_feature_names_out(['WorkoutType']),
                                         index=past_workouts.index)
    X_past_workouts = pd.concat([past_workouts.loc[mask_past, ['DistanceInMeters', 'TimeTotalInHours', 'HeartRateAverage']].copy(),
                                 past_workouts_encoded.loc[mask_past]], axis=1)
    X_past_workouts_imputed = imputer_y_hr.transform(X_past_workouts)
    X_past_workouts_poly = poly_y_hr.transform(X_past_workouts_imputed)
    past_workouts.loc[mask_past, 'EstimatedCalories'] = gb_model_y_hr.predict(X_past_workouts_poly)

    # Use the best model for future workouts with WorkoutType
    mask_future = future_workouts[['PlannedDistanceInMeters', 'PlannedDuration']].notna().all(axis=1)
    future_workouts_encoded = pd.DataFrame(one_hot_encoder.transform(future_workouts[['WorkoutType']]).toarray(),
                                           columns=one_hot_encoder.get_feature_names_out(['WorkoutType']),
                                           index=future_workouts.index)
    X_future_workouts = pd.concat([future_workouts.loc[mask_future, ['PlannedDistanceInMeters', 'PlannedDuration']].copy(),
                                   future_workouts_encoded.loc[mask_future]], axis=1)
    X_future_workouts_imputed = imputer_no_hr.transform(X_future_workouts)
    X_future_workouts_poly = poly_no_hr.transform(X_future_workouts_imputed)
    future_workouts.loc[mask_future, 'EstimatedCalories'] = xgb_model_no_hr.predict(X_future_workouts_poly)

    # Combine past and future workouts back together
    workouts_df = pd.concat([past_workouts, future_workouts])

    if best:
        return workouts_df,  {'rmse_gb_y_hr': rmse_gb_y_hr, 'rmse_xgb_no_hr': rmse_xgb_no_hr}
    else:
        return workouts_df, {
        'rmse_lr_y_hr': rmse_lr_y_hr, 'rmse_rf_y_hr': rmse_rf_y_hr, 'rmse_gb_y_hr': rmse_gb_y_hr,
        'rmse_lgb_y_hr': rmse_lgb_y_hr, 'rmse_xgb_y_hr': rmse_xgb_y_hr,
        'rmse_lr_no_hr': rmse_lr_no_hr, 'rmse_rf_no_hr': rmse_rf_no_hr, 'rmse_gb_no_hr': rmse_gb_no_hr,
        'rmse_lgb_no_hr': rmse_lgb_no_hr, 'rmse_xgb_no_hr': rmse_xgb_no_hr
    }


def estimate_calories_without_workout_type(activities_df, past_workouts, future_workouts, best):
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
    # Impute missing values for features with HeartRateAverage
    imputer_y_hr = KNNImputer()
    X_activities_y_hr_imputed = imputer_y_hr.fit_transform(X_activities_y_hr)

    # Impute missing values for features without HeartRateAverage
    imputer_no_hr = KNNImputer()
    X_activities_no_hr_imputed = imputer_no_hr.fit_transform(X_activities_no_hr)

    # Add polynomial features
    poly_y_hr = PolynomialFeatures(degree=2, include_bias=False)
    X_activities_y_hr_poly = poly_y_hr.fit_transform(X_activities_y_hr_imputed)
    poly_no_hr = PolynomialFeatures(degree=2, include_bias=False)
    X_activities_no_hr_poly = poly_no_hr.fit_transform(X_activities_no_hr_imputed)

    # Split data into training and test sets
    X_train_y_hr, X_test_y_hr, y_train, y_test = train_test_split(X_activities_y_hr_poly, y_activities, test_size=0.2, random_state=42)
    X_train_no_hr, X_test_no_hr, _, _ = train_test_split(X_activities_no_hr_poly, y_activities, test_size=0.2, random_state=42)

    if best:
        # Train and evaluate the models WITH HEART RATE
        # linear_model_y_hr, rmse_lr_y_hr = linear_regression_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
        # rf_model_y_hr, rmse_rf_y_hr = random_forest_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
        # gb_model_y_hr, rmse_gb_y_hr = gradient_boosting_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
        # lgb_model_y_hr, rmse_lgb_y_hr = lightgbm_model(X_train_y_hr, X_test_y_hr, y_train, y_test)
        xgb_model_y_hr, rmse_xgb_y_hr = xgboost_model(X_train_y_hr, X_test_y_hr, y_train, y_test)

        # Train and evaluate the models WITHOUT HEART RATE
        linear_model_no_hr, rmse_lr_no_hr = linear_regression_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
        # rf_model_no_hr, rmse_rf_no_hr = random_forest_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
        # gb_model_no_hr, rmse_gb_no_hr = gradient_boosting_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
        # lgb_model_no_hr, rmse_lgb_no_hr = lightgbm_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
        # xgb_model_no_hr, rmse_xgb_no_hr = xgboost_model(X_train_no_hr, X_test_no_hr, y_train, y_test)
    else:
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
    X_past_workouts_imputed = imputer_y_hr.transform(X_past_workouts)
    X_past_workouts_poly = poly_y_hr.transform(X_past_workouts_imputed)
    past_workouts.loc[mask_past, 'EstimatedCalories'] = xgb_model_y_hr.predict(X_past_workouts_poly)

    # Use the best model for future workouts without WorkoutType
    mask_future = future_workouts[['PlannedDistanceInMeters', 'PlannedDuration']].notna().all(axis=1)
    X_future_workouts = future_workouts.loc[mask_future, ['PlannedDistanceInMeters', 'PlannedDuration']].copy()
    X_future_workouts_imputed = imputer_no_hr.transform(X_future_workouts)
    X_future_workouts_poly = poly_no_hr.transform(X_future_workouts_imputed)
    future_workouts.loc[mask_future, 'EstimatedCalories'] =  linear_model_no_hr.predict(X_future_workouts_poly)

    # Combine past and future workouts back together
    workouts_df = pd.concat([past_workouts, future_workouts])

    if best:
        return workouts_df, {'rmse_xgb_y_hr': rmse_xgb_y_hr,'rmse_lr_no_hr': rmse_lr_no_hr}
    else:
        return workouts_df, {
        'rmse_lr_y_hr': rmse_lr_y_hr, 'rmse_rf_y_hr': rmse_rf_y_hr, 'rmse_gb_y_hr': rmse_gb_y_hr,
        'rmse_lgb_y_hr': rmse_lgb_y_hr, 'rmse_xgb_y_hr': rmse_xgb_y_hr,
        'rmse_lr_no_hr': rmse_lr_no_hr, 'rmse_rf_no_hr': rmse_rf_no_hr, 'rmse_gb_no_hr': rmse_gb_no_hr,
        'rmse_lgb_no_hr': rmse_lgb_no_hr, 'rmse_xgb_no_hr': rmse_xgb_no_hr
    }


"""
BEST PERFORMANCE
WITH WORKOUTTYPE
With Heart Rate:
Linear Regression: Improved from RMSE 69.09 to 63.75
Random Forest: Improved from RMSE 74.81 to 63.99
Gradient Boosting: Improved from RMSE 63.80 to 53.49 # THIS ONE
LIGHTGBM: Improved from RMSE 61.87 to 54.38
XGBOOST: Improved from RMSE 70.25 to 53.67

Without Heart Rate:
Linear Regression: Worsened from RMSE 84.73 to 87.77
Random Forest: Improved from RMSE 88.69 to 88.38
Gradient Boosting: Improved from RMSE 87.20 to 87.12
LIGHTGBM: Worsened from RMSE 77.05 to 87.06
XGBOOST: Improved from RMSE 82.22 to 83.58 #THIS ONE
"""


"""
WITHOUT WORKOUTTYPE
With Heart Rate:
Linear Regression: Improved from RMSE 79.34 to 66.93
Random Forest: Improved from RMSE 84.01 to 67.65
Gradient Boosting: Improved from RMSE 75.86 to 65.10
LIGHTGBM: Improved from RMSE 75.37 to 63.72
XGBOOST: Improved from RMSE 79.12 to 62.56 #THIS ONE!!!
Without Heart Rate:

Linear Regression: Improved from RMSE 98.45 to 88.35 #THIS ONE!!!
Random Forest: Improved from RMSE 96.13 to 93.03
Gradient Boosting: Improved from RMSE 98.35 to 94.26
LIGHTGBM: Worsened from RMSE 90.32 to 91.52
XGBOOST: Improved from RMSE 101.85 to 93.88

"""



"""
### VERSION 1

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
"""
