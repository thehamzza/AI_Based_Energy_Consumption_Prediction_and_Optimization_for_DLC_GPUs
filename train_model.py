# 📌 Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import joblib

# 📌 Step 2: Generate Synthetic Data (Realistic Variations & Anomalies)
np.random.seed(42)

timestamps = pd.date_range(start="2024-01-01", periods=1440, freq='h')

# Simulated weather patterns
temperature = 15 + 10 * np.sin(np.linspace(0, 6.28, len(timestamps))) + np.random.normal(0, 2, len(timestamps))
humidity = 50 + 10 * np.cos(np.linspace(0, 6.28, len(timestamps))) + np.random.normal(0, 5, len(timestamps))
wind_speed = np.random.uniform(5, 25, len(timestamps))
air_pressure = np.random.uniform(990, 1025, len(timestamps))
dew_point = temperature - np.random.uniform(2, 5, len(timestamps))
cloud_cover = np.abs(50 * np.sin(np.linspace(0, 6.28, len(timestamps)))) + np.random.normal(0, 10, len(timestamps))

# GPU-related features
gpu_utilization = np.clip(40 + 30 * np.sin(np.linspace(0, 12.56, len(timestamps))) + np.random.normal(0, 10, len(timestamps)), 0, 100)
fan_speed = np.clip(2000 + 500 * np.sin(np.linspace(0, 12.56, len(timestamps))) + np.random.normal(0, 200, len(timestamps)), 1000, 4000)
core_temperature = np.clip(50 + 10 * np.sin(np.linspace(0, 12.56, len(timestamps))) + np.random.normal(0, 5, len(timestamps)), 30, 90)
vram_utilization = np.clip(4000 + 1000 * np.sin(np.linspace(0, 6.28, len(timestamps))) + np.random.normal(0, 500, len(timestamps)), 1000, 8000)
power_limit = np.random.uniform(80, 120, len(timestamps))

# Workload Types (Categorical Encoding)
workload_encoded = np.random.choice([0, 1, 2, 3, 4], len(timestamps), p=[0.3, 0.3, 0.2, 0.1, 0.1])

# Time-based features
hour_of_day = timestamps.hour
day_of_week = timestamps.dayofweek
month_of_year = timestamps.month

# Simulating GPU Energy Consumption (Target Variable)
gpu_energy = ((150 + (gpu_utilization * 1.8) + (temperature * 0.6) - (fan_speed * 0.002)) + ((core_temperature * 0.9) + np.random.normal(0, 5, len(timestamps))))

# Injecting Anomalies Every ~200 Hours
for i in range(0, len(gpu_energy), 200):
    gpu_energy[i] += np.random.uniform(50, 100)

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'temperature': temperature,
    'humidity': humidity,
    'wind_speed': wind_speed,
    'air_pressure': air_pressure,
    'dew_point': dew_point,
    'cloud_cover': cloud_cover,
    'gpu_utilization': gpu_utilization,
    'fan_speed': fan_speed,
    'core_temperature': core_temperature,
    'vram_utilization': vram_utilization,
    'power_limit': power_limit,
    'workload_type': workload_encoded,
    'hour_of_day': hour_of_day,
    'day_of_week': day_of_week,
    'month_of_year': month_of_year,
    'gpu_energy': gpu_energy
})

df.set_index('timestamp', inplace=True)

# 📌 Step 3: Preprocess Data
features = df.columns[:-1]
target = 'gpu_energy'

scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

df_scaled = df.copy()
df_scaled[features] = scaler_features.fit_transform(df[features])
df_scaled[target] = scaler_target.fit_transform(df[[target]])

data_array = df_scaled[features.tolist() + [target]].values

SEQ_LENGTH = 24  # Use last 24 hours to predict next-hour GPU energy

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])
        y.append(data[i+seq_length, -1])
    return np.array(X), np.array(y)

X, y = create_sequences(data_array, SEQ_LENGTH)

# 📌 Step 4: Split Data for Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# 📌 Step 5: Train LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# 📌 Step 6: Save Model & Scaler
model.save("gpu_energy_model.h5")
joblib.dump(scaler_features, "scaler_features.pkl")
joblib.dump(scaler_target, "scaler_target.pkl")

print("Model training complete and saved!")