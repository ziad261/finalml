import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_and_tune_model(df, scaler=None, encoder=None):
    # ✅ Separate target before feature engineering
    y = df['Energy_Consumption_(kWh)']
    X = df.drop(columns=['Energy_Consumption_(kWh)', 'Date', 'Time', 'energy_level'], errors='ignore')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Grid Search
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    # ✅ Save model components
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rf_model.pkl")
    joblib.dump(list(X.columns), "models/feature_names.pkl")  # ✅ Save feature names used in training

    if scaler:
        joblib.dump(scaler, "models/scaler.pkl")
    if encoder:
        joblib.dump(encoder, "models/ordinal_encoder.pkl")

    return model, X_test, y_test
