# calorie_estimation_models.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from src.data_loader.files_extracting import FileLoader
from src.data_loader.files_saving import FileSaver
# this makes it so that the outputs of the predict methods have the id as a column
# instead of as the index
os.environ['NIXTLA_ID_AS_COL'] = '1'


from src.data_processing import *


# Modified Linear Regression function without preprocessing
def linear_regression_model(model_name, X_train, X_test, y_train, y_test):
    model = FileLoader().load_models(model_name)  # Try loading the saved model

    if model is None:  # If the model is not saved, train and save it
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        FileSaver().save_models(lr, model_name)
        model = lr  # Assign the newly trained model

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Linear Regression RMSE: {rmse:.4f}")
    return model, rmse


def ridge_regression_model(model_name, X_train, X_test, y_train, y_test):
    model = FileLoader().load_models(model_name)
    if model is None:
        param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        grid_search = GridSearchCV(
            Ridge(random_state=42),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        FileSaver().save_models(best_model, model_name)
        model = best_model
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Ridge Regression RMSE: {rmse:.4f} (alpha={model.alpha})")
    return model, rmse

def lasso_regression_model(model_name, X_train, X_test, y_train, y_test):
    model = FileLoader().load_models(model_name)
    if model is None:
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
        grid_search = GridSearchCV(
            Lasso(random_state=42, max_iter=10000),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        FileSaver().save_models(best_model, model_name)
        model = best_model
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Lasso Regression RMSE: {rmse:.4f} (alpha={model.alpha})")
    return model, rmse

def elasticnet_regression_model(model_name, X_train, X_test, y_train, y_test):
    model = FileLoader().load_models(model_name)
    if model is None:
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9]
        }
        grid_search = GridSearchCV(
            ElasticNet(random_state=42, max_iter=10000),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        FileSaver().save_models(best_model, model_name)
        model = best_model
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"ElasticNet Regression RMSE: {rmse:.4f} (alpha={model.alpha}, l1_ratio={model.l1_ratio})")
    return model, rmse


# Modified Random Forest function without preprocessing
def random_forest_model(model_name, X_train, X_test, y_train, y_test):
    model = FileLoader().load_models(model_name)  # Try loading the saved model

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
        FileSaver().save_models(best_model, model_name)
        model = best_model  # Assign the best model

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Random Forest RMSE: {rmse:.4f}")
    return model, rmse

# Modified Gradient Boosting function without preprocessing
def gradient_boosting_model(model_name, X_train, X_test, y_train, y_test):
    model = FileLoader().load_models(model_name)  # Try loading the saved model

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
        FileSaver().save_models(best_model, model_name)
        model = best_model  # Assign the best model

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Gradient Boosting RMSE: {rmse:.4f}")
    return model, rmse

# Modified LightGBM function without preprocessing
def lightgbm_model(model_name, X_train, X_test, y_train, y_test):
    model = FileLoader().load_models(model_name)  # Try loading the saved model

    if model is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 70],
            'feature_fraction': [0.6, 0.8, 1.0]
        }
        grid_search = GridSearchCV(
            lgb.LGBMRegressor(
                boosting_type='gbdt',
                num_threads=4,
                verbose=-1,
                force_col_wise=True,
                random_state=42,
                bagging_freq=5,          # Fixed based on best practices
                bagging_fraction=0.8,    # Fixed based on best practices
                lambda_l1=0.1,           # Fixed based on best practices
                lambda_l2=0.1,           # Fixed based on best practices
                min_child_samples=30     # Fixed based on best practices
            ),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        FileSaver().save_models(best_model, model_name)
        model = best_model  # Assign the best model

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"LightGBM RMSE: {rmse:.4f}")
    return model, rmse

# Modified XGBoost function without preprocessing
def xgboost_model(model_name, X_train, X_test, y_train, y_test):
    model = FileLoader().load_models(model_name)  # Try loading the saved model

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
        FileSaver().save_models(best_model, model_name)
        model = best_model  # Assign the best model

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
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


# Function to create preprocessing pipeline with PCA
def create_preprocessing_pipeline(use_poly=True, use_pca=False, n_components=None):
    steps = [
        ('preprocessor', get_preprocessor())
    ]

    if use_poly:
        steps.append(('poly', PolynomialFeatures(degree=2, include_bias=False)))

    steps.append(('scaler', StandardScaler()))  # Scaling before PCA

    if use_pca and n_components:
        steps.append(('pca', PCA(n_components=n_components)))

    preprocessing_pipeline = Pipeline(steps=steps)
    return preprocessing_pipeline


def transform_features(X, y, use_pca=False, n_components=None):
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessing_pipeline = FileLoader().load_models('preprocessing_pipeline')
    if preprocessing_pipeline is None:
        preprocessing_pipeline = create_preprocessing_pipeline(use_poly=True, use_pca=use_pca, n_components=n_components)
        # FileSaver().save_models(preprocessing_pipeline, 'preprocessing_pipeline')
    preprocessing_pipeline.fit(X_train_raw)
    FileSaver().save_models(preprocessing_pipeline, 'preprocessing_pipeline')

    X_train_transformed = preprocessing_pipeline.transform(X_train_raw)
    X_test_transformed = preprocessing_pipeline.transform(X_test_raw)

    return X_train_transformed, X_test_transformed, y_train, y_test


def estimate_calories_with_duration(features, target, use_pca=True, n_components=10, enable_regularization=True):

    X_train, X_test, y_train, y_test = transform_features(features, target, use_pca=use_pca, n_components=n_components)


    # Flags to enable/disable regularization models
    USE_RIDGE = True if enable_regularization else False
    USE_LASSO = True if enable_regularization else False
    USE_ELASTICNET = True if enable_regularization else False

    # Define model names and corresponding functions
    models = [
        ("Linear Regression", linear_regression_model),
        ("Random Forest", random_forest_model),
        ("Gradient Boosting", gradient_boosting_model),
        ("LightGBM", lightgbm_model),
        ("XGBoost", xgboost_model)
    ]

    # Add regularized models based on flags
    if USE_RIDGE:
        models.append(("Ridge Regression", ridge_regression_model))
    if USE_LASSO:
        models.append(("Lasso Regression", lasso_regression_model))
    if USE_ELASTICNET:
        models.append(("ElasticNet Regression", elasticnet_regression_model))

    def create_model_configs(models, X_train, X_test):
        configs = []
        for model_name, model_func in models:
            configs.append({
                "name": f"{model_name} with Duration with WorkoutType{' + PCA' if use_pca else ''}",
                "X_train": X_train,
                "X_test": X_test,
                "model_func": model_func
            })

        return configs

    # Create model configurations
    model_configs = create_model_configs(models, X_train, X_test)

    # Iterate through each model configuration and train
    for config in model_configs:
        model, rmse = train_and_evaluate(config["name"], config["X_train"], config["X_test"], y_train, y_test, config["model_func"])
        config['model'] = model
        config['rmse'] = rmse


    return model_configs





### NIXTLA ###


def prepare_time_series_data(df, unique_id='series_1'):
    """
    Prepare the DataFrame for time series forecasting.
    Ensures 'Date' column exists, includes all dates, and incorporates exogenous variables.

    Parameters:
    - df: DataFrame with data (columns: 'Date', 'TotalDuration', 'WorkoutType', optionally 'Calories')
    - unique_id: Identifier for the time series

    Returns:
    - ts_data: DataFrame ready for StatsForecast with 'unique_id', 'ds', 'y' (if present), and exogenous variables
    """

    if 'Date' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'Date'})


    # Create a complete date range from the earliest to the latest date
    full_date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    df_full = pd.DataFrame({'Date': full_date_range})

    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df_full['Date'] = pd.to_datetime(df_full['Date']).dt.normalize()

    # Merge with the original data to include all dates
    df_full = df_full.merge(df, on='Date', how='left')

    # Fill missing 'TotalDuration' with 0 (no workout)
    df_full['TotalDuration'] = df_full['TotalDuration'].fillna(0)

    # Fill missing 'WorkoutType' with 'None' (no workout)
    df_full['WorkoutType'] = df_full['WorkoutType'].fillna('None')

    # Define workout types for one-hot encoding, ensuring 'None' is included
    workout_types = ['None', 'Bike', 'Run', 'Swim']
    df_full['WorkoutType'] = pd.Categorical(df_full['WorkoutType'], categories=workout_types)


    # Return transformed DataFrame without creating dummies here, as this will be handled in the pipeline
    ts_data = pd.DataFrame({
        'unique_id': unique_id,
        'ds': df_full['Date'],
        'TotalDuration': df_full['TotalDuration'],
        'WorkoutType': df_full['WorkoutType']
    })

    if 'Calories' in df_full.columns:
        ts_data['y'] = df_full['Calories'].fillna(0)

    # **Workaround: Shift 'TotalDuration' to avoid collinearity with trend**
    ts_data['TotalDuration'] = ts_data['TotalDuration'] + 0.01  # Adjust the constant as needed otherwise ValueError: xreg is rank deficient

    return ts_data


def estimate_calories_with_nixtla(features, target, future_w_df, unique_id='series_1'):
    """
    Estimate calories using StatsForecast with exogenous variables.

    Parameters:
    - features: DataFrame containing historical features (including 'TotalDuration' and 'WorkoutType')
    - target: Series containing target variable ('Calories')
    - future_w_df: DataFrame containing future workouts (columns: 'Date', 'TotalDuration', 'WorkoutType')
    - unique_id: Identifier for the time series

    Returns:
    - forecast: DataFrame containing forecasted calories with 'ds' and 'y'
    - sf: Trained StatsForecast model
    """
    # Combine features and target into a single DataFrame
    df = features.copy()
    df['Calories'] = target


    # Define your preprocessing pipeline
    preprocessing_pipeline = ColumnTransformer(
        transformers=[
            ('one_hot', OneHotEncoder(categories=[['Bike', 'Run', 'Swim']], sparse_output=False, handle_unknown='ignore'), ['WorkoutType']),
            ('scaler', StandardScaler(), ['TotalDuration'])  # Apply scaler to TotalDuration
        ],
        remainder='passthrough'  # Keep the other columns
    )


    # Prepare historical data
    train = prepare_time_series_data(df, unique_id=unique_id)

    # Prepare future data (exclude 'Calories')
    X_test = prepare_time_series_data(future_w_df, unique_id=unique_id)


    # Fit and transform your training data
    train_transformed = preprocessing_pipeline.fit_transform(train[['TotalDuration','WorkoutType']])

    # Create a DataFrame from the transformed data
    train_transformed_df = pd.DataFrame(train_transformed)

    # Add back the 'unique_id', 'ds', and 'y' columns from the original train DataFrame
    train_transformed_df['unique_id'] = train['unique_id'].values
    train_transformed_df['ds'] = train['ds'].values
    train_transformed_df['y'] = train['y'].values

    # For test data, apply the same pipeline
    X_test_transformed = preprocessing_pipeline.transform(X_test[['TotalDuration','WorkoutType']])

    # Create a DataFrame from the transformed test data
    X_test_transformed_df = pd.DataFrame(X_test_transformed)

    # Add back the 'unique_id' and 'ds' columns from the original X_test DataFrame
    X_test_transformed_df['unique_id'] = X_test['unique_id'].values
    X_test_transformed_df['ds'] = X_test['ds'].values

    # At this point, train_transformed_df and X_test_transformed_df are ready for use.

    # Initialize the StatsForecast model with AutoARIMA
    models = [AutoARIMA(season_length=7)]  # Weekly seasonality
    sf = StatsForecast(models=models, freq='D', n_jobs=-1)

    # Define forecasting parameters
    horizon = X_test.shape[0]
    level = [95]

    # Perform the forecast using the specified method
    fcst = sf.forecast(df=train_transformed_df, h=horizon, X_df=X_test_transformed_df, level=level)


    # **Handling Non-Workout Days**: Ensure that days without workouts have Calories = 0
    # Identify which days have workouts based on 'TotalDuration' in X_test
    # has_workout = X_test['TotalDuration'] == 0.01  # Considering the shift applied earlier

    # Set Calories to 0 for days without workouts
    #fcst.loc[~has_workout, 'AutoARIMA'] = 0 # NOTE: STILL DON'T KNOW WHAT TO DO HERE
    FileSaver().save_models(sf, "statsforecast_model")


    return fcst, sf


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




PREVIOUS WITHOUT PCA

Performance Metrics:
Linear Regression with Duration with WorkoutType RMSE: 90.44261279868016
Random Forest with Duration with WorkoutType RMSE: 92.35009718944328
Gradient Boosting with Duration with WorkoutType RMSE: 92.52689118300806
LightGBM with Duration with WorkoutType RMSE: 93.62855437802837
XGBoost with Duration with WorkoutType RMSE: 91.26319865609005

WITH PCA

Performance Metrics:
Linear Regression with Duration with WorkoutType + PCA RMSE: 89.18575651184022
Random Forest with Duration with WorkoutType + PCA RMSE: 92.64961465932079
Gradient Boosting with Duration with WorkoutType + PCA RMSE: 88.9923076391858
LightGBM with Duration with WorkoutType + PCA RMSE: 93.59738060946677
XGBoost with Duration with WorkoutType + PCA RMSE: 89.14774342763234
Ridge Regression with Duration with WorkoutType + PCA RMSE: 89.24537933176896
Lasso Regression with Duration with WorkoutType + PCA RMSE: 89.24908379231046
ElasticNet Regression with Duration with WorkoutType + PCA RMSE: 89.4404476536283

"""
