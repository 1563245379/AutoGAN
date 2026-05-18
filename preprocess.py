import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

INPUT_PATH = "data/raw_data.csv"
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
TRAIN_LABEL_PATH = "data/train_label.csv"
TEST_LABEL_PATH = "data/test_label.csv"

deleted_cols = [
    'site_id',
    'building_id',
    'air_temperature_mean_lag7',
    'air_temperature_max_lag7',
    'air_temperature_min_lag7',
    'air_temperature_std_lag7',
    'air_temperature_mean_lag73',
    'air_temperature_max_lag73',
    'air_temperature_min_lag73',
    'air_temperature_std_lag73',
    'hour',
    'weekday',
    'month',
    'year',
    'hour_x',
    'hour_y',
    'month_x',
    'month_y',
    'weekday_x',
    'weekday_y',
    'gte_hour',
    'gte_weekday',
    'gte_month',
    'gte_building_id',
    'gte_primary_use',
    'gte_site_id',
    'gte_meter',
    'gte_meter_hour',
    'gte_meter_weekday',
    'gte_meter_month',
    'gte_meter_building_id',
    'gte_meter_primary_use',
    'gte_meter_site_id',
    'gte_meter_building_id_hour',
    'gte_meter_building_id_weekday',
    'gte_meter_building_id_month',
]

df = pd.read_csv(INPUT_PATH)
print(f"Original: {df.shape}")

df = df.dropna()

building_id_ref = df['building_id'].copy()
timestamp_ref = pd.to_datetime(df['timestamp'])

object_cols = df.select_dtypes(include="object").columns.tolist()
df = df.drop(columns=object_cols)

df = df.drop(columns=deleted_cols)

df = df.reset_index(drop=True)
df.insert(0, 'id', df.index)

feature_cols = [c for c in df.columns if c not in ('id', 'anomaly')]
scaler = MinMaxScaler(feature_range=(-1, 1))
df[feature_cols] = scaler.fit_transform(df[feature_cols])

df['building_id'] = building_id_ref.values
df['timestamp'] = timestamp_ref.values
df = df.sort_values(['building_id', 'timestamp']).reset_index(drop=True)

df['date'] = df['timestamp'].dt.date
df['meterReadings_daily_std'] = df.groupby(['building_id', 'date'])['meter_reading'].transform('std')

df_indexed = df.set_index('timestamp')
df['air_temperature_std_lag7'] = (
    df_indexed.groupby('building_id')['air_temperature']
    .rolling('7D').std()
    .reset_index(level=0, drop=True)
    .values
)
df['air_temperature_std_lag73'] = (
    df_indexed.groupby('building_id')['air_temperature']
    .rolling('73D').std()
    .reset_index(level=0, drop=True)
    .values
)

df = df.drop(columns=['building_id', 'timestamp', 'date'])

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
