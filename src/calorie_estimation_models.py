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
from sklearn.compose import ColumnTransformer

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


# Modified Linear Regression function without preprocessing
def linear_regression_model(model_name, X_train, X_test, y_train, y_test):
    model = load_model(model_name)  # Try loading the saved model

    if model is None:  # If the model is not saved, train and save it
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        save_model(lr, model_name)  # Save the trained model
        model = lr  # Assign the newly trained model

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Linear Regression RMSE: {rmse:.4f}")
    return model, rmse

# Modified Random Forest function without preprocessing
def random_forest_model(model_name, X_train, X_test, y_train, y_test):
    model = load_model(model_name)  # Try loading the saved model

    if model is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        save_model(best_model, model_name)  # Save the model
        model = best_model  # Assign the best model

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Random Forest RMSE: {rmse:.4f}")
    return model, rmse

# Modified Gradient Boosting function without preprocessing
def gradient_boosting_model(model_name, X_train, X_test, y_train, y_test):
    model = load_model(model_name)  # Try loading the saved model

    if model is None:
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        grid_search = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        save_model(best_model, model_name)
        model = best_model  # Assign the best model

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Gradient Boosting RMSE: {rmse:.4f}")
    return model, rmse

# Modified LightGBM function without preprocessing
def lightgbm_model(model_name, X_train, X_test, y_train, y_test):
    model = load_model(model_name)  # Try loading the saved model

    if model is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'learning_rate': [0.01, 0.05, 0.1]
        }
        grid_search = GridSearchCV(
            lgb.LGBMRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        save_model(best_model, model_name)
        model = best_model  # Assign the best model

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"LightGBM RMSE: {rmse:.4f}")
    return model, rmse

# Modified XGBoost function without preprocessing
def xgboost_model(model_name, X_train, X_test, y_train, y_test):
    model = load_model(model_name)  # Try loading the saved model

    if model is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        grid_search = GridSearchCV(
            xgb.XGBRegressor(random_state=42, objective='reg:squarederror'),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        save_model(best_model, model_name)
        model = best_model  # Assign the best model

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"XGBoost RMSE: {rmse:.4f}")
    return model, rmse


# Define a function to handle training and evaluation
def train_and_evaluate(model_name, X_train, X_test, y_train, y_test, model_func):
    model, rmse = model_func(model_name, X_train, X_test, y_train, y_test)
    print(f"{model_name} RMSE: {rmse}")
    return model, rmse


def get_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('one_hot', OneHotEncoder(categories=[['Bike', 'Run', 'Swim']], sparse_output=False, handle_unknown='ignore'), ['WorkoutType']),
            ('imputer', KNNImputer(), ['TotalDuration'])
        ],
        remainder='drop'
    )
    return preprocessor


def create_preprocessing_pipeline(use_poly = True):
    steps = [
        ('preprocessor', get_preprocessor())
        ]
    if use_poly:
        steps.append(
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        )
    steps.append(('scaler', StandardScaler()))
    preprocessing_pipeline = Pipeline(steps=steps)
    return preprocessing_pipeline


def transform_features(X, y):
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessing_pipeline = create_preprocessing_pipeline()
    preprocessing_pipeline.fit(X_train_raw)

    X_train_transformed = preprocessing_pipeline.transform(X_train_raw)
    X_test_transformed = preprocessing_pipeline.transform(X_test_raw)

    return preprocessing_pipeline, X_train_transformed, X_test_transformed, y_train, y_test


def estimate_calories_with_duration(X, y):
    preprocessing_pipeline, X_train, X_test, y_train, y_test = transform_features(X, y)
    save_model(preprocessing_pipeline, 'preprocessing_pipeline')

    def create_model_configs(models, X_train, X_test):
        configs = []
        for model_name, model_func in models:
            configs.append({
                "name": f"{model_name} with Duration with WorkoutType",
                "X_train": X_train,
                "X_test": X_test,
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
    model_configs = create_model_configs(models, X_train, X_test)

    # Iterate through each model configuration and train
    for config in model_configs:
        model, rmse = train_and_evaluate(config["name"], config["X_train"], config["X_test"], y_train, y_test, config["model_func"])
        config['model'] = model
        config['rmse'] = rmse








###################


"""
Model Performance Results

| Model               | With Duration & WorkoutType | With WorkoutType & HR | With WorkoutType & without HR | Without WorkoutType & HR | Without WorkoutType & without HR |
|---------------------|-----------------------------|-----------------------|-------------------------------|--------------------------|----------------------------------|
| Linear Regression    | 89.19                       | 63.75                 | 87.77                         | 66.93                    | 88.35                            |
| Random Forest        | 92.34                       | 63.99                 | 88.38                         | 67.65                    | 93.03                            |
| Gradient Boosting    | 91.99                       | 53.49                 | 87.12                         | 65.10                    | 94.26                            |
| LightGBM             | 91.05                       | 54.38                 | 87.06                         | 63.72                    | 91.52                            |
| XGBoost              | 91.26                       | 53.67                 | 83.58                         | 62.56                    | 93.88                            |
"""

"""

Performance Metrics WITH DURATION & WORKOUTTYPE without poly:
Linear Regression with Duration with WorkoutTYpe RMSE: 99.89309317090766
Random Forest with Duration with WorkoutTYpe RMSE: 92.06899183202154
Gradient Boosting with Duration with WorkoutTYpe RMSE: 90.97004409283143
LightGBM with Duration with WorkoutTYpe RMSE: 90.54056355551425
XGBoost with Duration with WorkoutTYpe RMSE: 91.37666255801469
"""
