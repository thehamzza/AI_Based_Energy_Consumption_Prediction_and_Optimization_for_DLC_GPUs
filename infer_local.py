import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# Load Model & Scalers
model = load_model("gpu_energy_model.h5")
scaler_features = joblib.load("scaler_features.pkl")  # Feature Scaler
scaler_target = joblib.load("scaler_target.pkl")  # Target Scaler

# Generate Synthetic Future Data (24 Timestamps)
future_timestamps = pd.date_range(start="2025-01-01", periods=24, freq='h')

# Generate All 15 Features (Simulated)
future_data = pd.DataFrame({
    "temperature": np.random.uniform(0, 35, len(future_timestamps)),
    "humidity": np.random.uniform(20, 80, len(future_timestamps)),
    "wind_speed": np.random.uniform(0, 20, len(future_timestamps)),
    "air_pressure": np.random.uniform(990, 1025, len(future_timestamps)),
    "dew_point": np.random.uniform(5, 15, len(future_timestamps)),
    "cloud_cover": np.random.uniform(0, 100, len(future_timestamps)),
    "gpu_utilization": np.random.uniform(0, 100, len(future_timestamps)),
    "fan_speed": np.random.uniform(1000, 4000, len(future_timestamps)),
    "core_temperature": np.random.uniform(30, 90, len(future_timestamps)),
    "vram_utilization": np.random.uniform(1000, 8000, len(future_timestamps)),
    "power_limit": np.random.uniform(80, 120, len(future_timestamps)),
    "workload_type": np.random.choice([0, 1, 2, 3, 4], len(future_timestamps)),  
    "hour_of_day": future_timestamps.hour,
    "day_of_week": future_timestamps.dayofweek,
    "month_of_year": future_timestamps.month
}, index=future_timestamps)

# Scale Data Using Trained Scaler
future_data_scaled = scaler_features.transform(future_data)

# Ensure Correct LSTM Input Shape: (batch_size=1, SEQ_LENGTH=24, num_features=15)
X_future = np.array([future_data_scaled])  
X_future = X_future.reshape(1, 24, 15)  # Match input shape used during training

# Make Prediction
prediction = model.predict(X_future)
predicted_energy = scaler_target.inverse_transform([[prediction[0][0]]])[0][0]

print(f"Predicted GPU Energy Consumption: {predicted_energy:.2f} Watts")

# 📌 Plot Predictions
plt.figure(figsize=(12, 5))
plt.plot(future_data.index, future_data["gpu_utilization"], label="GPU Utilization (%)", color='blue')
plt.axhline(predicted_energy, color='red', linestyle='--', label="Predicted Energy Next Hour")
plt.xlabel("Timestamp")
plt.ylabel("GPU Energy Consumption (Watts)")
plt.title("LSTM Model: GPU Energy Consumption Prediction")
plt.legend()
plt.grid()
plt.show()