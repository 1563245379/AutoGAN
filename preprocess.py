import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

INPUT_PATH = "data/raw_data.csv"
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
TRAIN_LABEL_PATH = "data/train_label.csv"
TEST_LABEL_PATH = "data/test_label.csv"

deleted_cols = [
    'gte_meter_hour',
    'gte_meter_weekday',
    'gte_meter_building_id',
    'gte_meter_primary_use',
    'gte_meter_site_id',
    'gte_meter_building_id_hour',
    'gte_meter_building_id_weekday',
    'gte_meter_building_id_month',
    'air_temperature_mean_lag73',
    'air_temperature_max_lag73',
    'air_temperature_min_lag73',
    'air_temperature_mean_lag7',
    'air_temperature_max_lag7',
    'air_temperature_min_lag7',
    'gte_meter'
]

df = pd.read_csv(INPUT_PATH)
print(f"Original: {df.shape}")

df['timestamp'] = pd.to_datetime(df['timestamp'])

ints = []
for col in df.columns:
    if df[col].dtype == int:
        ints.append(col)

df = df.drop(columns=ints[9:])

object_cols = df.select_dtypes(include="object").columns.tolist()
df = df.drop(columns=object_cols)
df = df.drop(columns=deleted_cols)

df['meter_reading'] = df.groupby('building_id')['meter_reading'].transform(lambda x: x.fillna(x.mean()))

df['date'] = df['timestamp'].dt.date
df['meterReadings_daily_std'] = df.groupby(['building_id', 'date'])['meter_reading'].transform(lambda x: x.std(ddof=0))

df = df.sort_values(['building_id', 'timestamp']).reset_index(drop=True)
df.insert(0, 'id', df.index)

shifts = [1, 6, 12, 24, 2*24, 3*24, 4*24, 7*24]
for s in shifts:
    df[f'lag_value_{s}'] = np.nan
    df[f'lag_value_{-s}'] = np.nan
    for bid, idx in df.groupby('building_id').groups.items():
        group = df.loc[idx].set_index('timestamp').sort_index()
        full_range = pd.date_range(group.index.min(), group.index.max(), freq='h')
        mean_val = group['meter_reading'].mean()
        reading = group['meter_reading'].reindex(full_range).fillna(mean_val)

        for s in shifts:
            df.loc[idx, f'lag_value_{s}'] = reading.shift(s).reindex(group.index).fillna(mean_val).values
            df.loc[idx, f'lag_value_{-s}'] = reading.shift(-s).reindex(group.index).fillna(mean_val).values

for col in df.columns:
    k = df[col].isnull().sum()
    if k > 0:
        print(f'{col} : {k}')

df = df.drop(columns=['timestamp', 'date'])

feature_cols = [c for c in df.columns if c not in ('id', 'anomaly')]
standard_scaler = StandardScaler()
df[feature_cols] = standard_scaler.fit_transform(df[feature_cols])
minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
df[feature_cols] = minmax_scaler.fit_transform(df[feature_cols])

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = pd.DataFrame(train_df)
test_df = pd.DataFrame(test_df)

train_label = train_df[['id', 'anomaly']]
test_label = test_df[['id', 'anomaly']]
train_df = train_df.drop(columns=['anomaly'])
test_df = test_df.drop(columns=['anomaly'])

print(f"Train: {train_df.shape}")
print(f"Test: {test_df.shape}")

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)
train_label.to_csv(TRAIN_LABEL_PATH, index=False)
test_label.to_csv(TEST_LABEL_PATH, index=False)
