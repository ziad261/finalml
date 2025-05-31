import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])

    # Avoid chained assignment to prevent FutureWarning
    df['Appliance Type'] = df['Appliance Type'].fillna(df['Appliance Type'].mode()[0])
    df['Energy_Consumption_(kWh)'] = df['Energy_Consumption_(kWh)'].fillna(df['Energy_Consumption_(kWh)'].median())

    df = df.drop_duplicates()

    return df

def scale_features(df):
    scaler = StandardScaler()
    df[['Outdoor Temperature (°C)']] = scaler.fit_transform(
        df[[ 'Outdoor Temperature (°C)']]
    )
    return df, scaler
df = pd.read_csv(r'C:\Users\lenovo\Desktop\final ml\smart_home_energy_modified.csv', parse_dates=['Date'])
print(df.columns.tolist())
