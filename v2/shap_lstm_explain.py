import os
import json
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Output folder for results
BASE_OUTPUT_DIR = "feature_analysis_results"
SHAP_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "shap_lstm_outputs")
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

# Feature keys used in LSTM model
FEATURE_KEYS = [
    'ambientTemperature', 'powerConsumption', 'workLoadType',
    'hour_of_day', 'day_of_week', 'month_of_year',
    'device_on', 'open_duration', 'power_prev_hour',
    'temp_prev_hour', 'duration_prev_hour',
    'rolling_power_mean_3h', 'rolling_temp_mean_3h', 'rolling_duration_mean_3h',
    'rolling_power_std_3h', 'rolling_temp_std_3h', 'rolling_duration_std_3h',
    'power_diff', 'temp_diff', 'duration_diff',
    'is_weekend', 'is_night'
]

# Load trained model and scaler
model = load_model("lstm_device_energy_model.h5")
with open("lstm_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load enhanced dataset
with open("enhanced_synthetic_data.json") as f:
    raw = json.load(f)

# Build 3-step sequences
samples = []
for device_data in raw.values():
    df = pd.DataFrame(device_data)
    df.sort_values("timestamp", inplace=True)
    if not all(k in df.columns for k in FEATURE_KEYS):
        continue
    matrix = df[FEATURE_KEYS].fillna(-999).values
    if len(matrix) >= 6:
        for i in range(len(matrix) - 2):
            samples.append(matrix[i:i+3])
    if len(samples) >= 60:
        break

X_seq = np.array(samples)
if len(X_seq) < 20:
    print("Not enough sequences for SHAP. Found:", len(X_seq))
    exit()

# Scale the sequences
X_flat = X_seq.reshape(-1, X_seq.shape[-1])
X_scaled = scaler.transform(X_flat).reshape(X_seq.shape)
X_flatseq = X_scaled.reshape(X_scaled.shape[0], -1)  # (samples, 66)

# Model wrapper for KernelExplainer
def lstm_predict_flat(flat_X):
    reshaped = flat_X.reshape(flat_X.shape[0], 3, len(FEATURE_KEYS))
    return model.predict(reshaped)

# Create SHAP explainer
explainer = shap.KernelExplainer(lstm_predict_flat, X_flatseq[:10])
shap_values = explainer.shap_values(X_flatseq[10:15], nsamples=100)

# Generate feature names for all 3 time steps
feature_names_flat = [f"{feat}_t{i}" for i in range(1, 4) for feat in FEATURE_KEYS]
output_names = ["power", "temperature", "duration"]

# Plot each SHAP output
for i in range(3):
    sv = shap_values[:, :, i]  # shape (samples, 66)

    if sv.shape[1] != len(feature_names_flat):
        print(f"Skipping {output_names[i]}: SHAP shape mismatch. SHAP shape = {sv.shape}")
        continue

    shap.summary_plot(
        sv,
        X_flatseq[10:15],
        feature_names=feature_names_flat,
        show=False
    )
    plt.title(f"SHAP Summary for {output_names[i]}")
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_OUTPUT_DIR, f"shap_lstm_{output_names[i]}.png"), dpi=300)
    plt.close()

print("SHAP summary plots saved to:", SHAP_OUTPUT_DIR)