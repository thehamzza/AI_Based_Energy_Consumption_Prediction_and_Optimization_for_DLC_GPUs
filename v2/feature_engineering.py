import json
import pandas as pd
import numpy as np

def load_data(filepaths):
    all_data = []
    for path in filepaths:
        print(f"Loading file: {path}")
        with open(path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            device_type = 'gpu' if 'gpu' in path.lower() else 'refrigerator'
            for record in data:
                record['device_id'] = record.get('device_id', device_type)

                # Normalize temperature column
                if 'core_temperature' in record:
                    record['temperature'] = record['core_temperature']
                elif 'internal_temperature' in record:
                    record['temperature'] = record['internal_temperature']

                all_data.append(record)
            print(f"  → Loaded {len(data)} records (flat list)")

        elif isinstance(data, dict):
            record_count = 0
            for device_id, records in data.items():
                for record in records:
                    record['device_id'] = device_id

                    if 'core_temperature' in record:
                        record['temperature'] = record['core_temperature']
                    elif 'internal_temperature' in record:
                        record['temperature'] = record['internal_temperature']

                    all_data.append(record)
                    record_count += 1
            print(f"  → Loaded {record_count} records (grouped by device_id)")
        else:
            print(f"Unsupported data format in {path}")

    return pd.DataFrame(all_data)

def add_temporal_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    if 'doorOpen' in df.columns:
        df['doorOpen'] = pd.to_datetime(df['doorOpen'], errors='coerce', utc=True).dt.tz_convert(None)
    else:
        df['doorOpen'] = pd.NaT

    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month_of_year'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour_of_day'].between(0, 6).astype(int)
    df['device_on'] = df.get('powerConsumption', 0) > 5

    # Door open duration in minutes
    df['open_duration'] = (
        (df['timestamp'] - df['doorOpen']).dt.total_seconds() / 60
    ).clip(lower=0).fillna(0)

    df = df.sort_values(by=['device_id', 'timestamp'])

    df['power_prev_hour'] = df.groupby('device_id')['powerConsumption'].shift(1)
    df['temp_prev_hour'] = df.groupby('device_id')['temperature'].shift(1)
    df['duration_prev_hour'] = df.groupby('device_id')['open_duration'].shift(1)

    df['rolling_power_mean_3h'] = df.groupby('device_id')['powerConsumption'].rolling(3).mean().reset_index(0, drop=True)
    df['rolling_temp_mean_3h'] = df.groupby('device_id')['temperature'].rolling(3).mean().reset_index(0, drop=True)
    df['rolling_duration_mean_3h'] = df.groupby('device_id')['open_duration'].rolling(3).mean().reset_index(0, drop=True)

    df['rolling_power_std_3h'] = df.groupby('device_id')['powerConsumption'].rolling(3).std().reset_index(0, drop=True)
    df['rolling_temp_std_3h'] = df.groupby('device_id')['temperature'].rolling(3).std().reset_index(0, drop=True)
    df['rolling_duration_std_3h'] = df.groupby('device_id')['open_duration'].rolling(3).std().reset_index(0, drop=True)

    df['power_diff'] = df['powerConsumption'] - df['power_prev_hour']
    df['temp_diff'] = df['temperature'] - df['temp_prev_hour']
    df['duration_diff'] = df['open_duration'] - df['duration_prev_hour']

    return df

def save_as_json(df, filename):
    # Convert all Timestamp columns to ISO format strings
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].astype(str)
    if 'doorOpen' in df.columns:
        df['doorOpen'] = df['doorOpen'].astype(str)

    grouped = df.groupby('device_id')
    output = {}
    for device_id, group in grouped:
        group_sorted = group.sort_values(by='timestamp')
        output[device_id] = group_sorted.to_dict(orient='records')

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved enhanced data to {filename}")

if __name__ == '__main__':
    input_files = [
        'gpu_synthetic_data.json',
        'refrigerator_synthetic_data.json'
    ]

    df = load_data(input_files)
    print(f"Total combined records: {df.shape}")

    df = add_temporal_features(df)

    # Fill NaNs with placeholder, not dropping them
    df = df.fillna(-999).reset_index(drop=True)
    print(f"Final enhanced dataset shape: {df.shape}")

    save_as_json(df, 'enhanced_synthetic_data.json')