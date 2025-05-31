import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def add_time_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['week'] = df['Date'].dt.isocalendar().week

    df['hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
    df['minute'] = pd.to_datetime(df['Time'], format='%H:%M').dt.minute

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df

def encode_features(df):
    cols_to_encode = ['Appliance Type', 'Season']
    oe = OrdinalEncoder()
    df[cols_to_encode] = oe.fit_transform(df[cols_to_encode])
    return df, oe

def engineer_features(df):
    df = add_time_features(df)

    # Create 'energy_level' before encoding
    df['energy_level'] = pd.qcut(df['Energy_Consumption_(kWh)'], q=3, labels=['Low', 'Medium', 'High'])

    df, oe = encode_features(df)

    X = df.drop(columns=['Energy_Consumption_(kWh)', 'Date', 'Time','energy_level'])
    y = df['Energy_Consumption_(kWh)']

    return X, y, oe
