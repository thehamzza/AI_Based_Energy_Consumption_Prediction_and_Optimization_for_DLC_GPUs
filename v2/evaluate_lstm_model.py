# import os
# import json
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import load_model

# # Output directory
# OUTPUT_FOLDER = 'evaluate_lstm_model'
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# FEATURE_KEYS = [
#     'ambientTemperature', 'powerConsumption', 'workLoadType',
#     'hour_of_day', 'day_of_week', 'month_of_year',
#     'device_on', 'open_duration', 'power_prev_hour',
#     'temp_prev_hour', 'duration_prev_hour',
#     'rolling_power_mean_3h', 'rolling_temp_mean_3h', 'rolling_duration_mean_3h',
#     'rolling_power_std_3h', 'rolling_temp_std_3h', 'rolling_duration_std_3h',
#     'power_diff', 'temp_diff', 'duration_diff',
#     'is_weekend', 'is_night'
# ]

# TARGET_KEYS = ['powerConsumption', 'temperature', 'temperature_duration']

# def load_enhanced_sequence_data(path='enhanced_synthetic_data.json', seq_length=3):
#     with open(path, 'r') as f:
#         raw_data = json.load(f)

#     sequences = []
#     labels = []

#     for device_id, records in raw_data.items():
#         df = pd.DataFrame(records)
#         if len(df) < seq_length + 1:
#             continue

#         df = df.sort_values('timestamp')
#         df = df.fillna(-999)

#         feature_data = df[FEATURE_KEYS].values
#         target_data = df[TARGET_KEYS].values

#         for i in range(len(df) - seq_length):
#             seq = feature_data[i:i + seq_length]
#             label = target_data[i + seq_length]
#             sequences.append(seq)
#             labels.append(label)

#     return np.array(sequences), np.array(labels)

# def save_metrics(y_true, y_pred, name):
#     for i, key in enumerate(TARGET_KEYS):
#         yt = y_true[:, i]
#         yp = y_pred[:, i]

#         mae = mean_absolute_error(yt, yp)
#         rmse = np.sqrt(mean_squared_error(yt, yp))
#         r2 = r2_score(yt, yp)

#         with open(os.path.join(OUTPUT_FOLDER, f"{name}_{key}_metrics.txt"), 'w') as f:
#             f.write(f"{name} - {key} Evaluation\n")
#             f.write(f"MAE: {mae:.4f}\n")
#             f.write(f"RMSE: {rmse:.4f}\n")
#             f.write(f"R²: {r2:.4f}\n")

# def plot_pred_vs_actual(y_true, y_pred, name):
#     for i, key in enumerate(TARGET_KEYS):
#         plt.figure(figsize=(8, 6))
#         plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
#         plt.plot([y_true[:, i].min(), y_true[:, i].max()], [y_true[:, i].min(), y_true[:, i].max()], 'r--')
#         plt.xlabel("Actual")
#         plt.ylabel("Predicted")
#         plt.title(f"{name} - {key} - Predicted vs Actual")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUTPUT_FOLDER, f"{name}_{key}_pred_vs_actual.png"))
#         plt.close()

# def plot_residuals(y_true, y_pred, name):
#     for i, key in enumerate(TARGET_KEYS):
#         residuals = y_true[:, i] - y_pred[:, i]
#         plt.figure(figsize=(8, 6))
#         plt.scatter(range(len(residuals)), residuals, alpha=0.6)
#         plt.axhline(0, color='red', linestyle='--')
#         plt.xlabel("Sample Index")
#         plt.ylabel("Residual")
#         plt.title(f"{name} - {key} - Residuals")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUTPUT_FOLDER, f"{name}_{key}_residuals.png"))
#         plt.close()

# def plot_error_dist(y_true, y_pred, name):
#     for i, key in enumerate(TARGET_KEYS):
#         errors = y_true[:, i] - y_pred[:, i]
#         plt.figure(figsize=(8, 6))
#         plt.hist(errors, bins=30, alpha=0.7)
#         plt.xlabel("Error")
#         plt.ylabel("Frequency")
#         plt.title(f"{name} - {key} - Error Distribution")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUTPUT_FOLDER, f"{name}_{key}_error_distribution.png"))
#         plt.close()

# def evaluate_lstm_model():
#     X, y = load_enhanced_sequence_data(seq_length=3)

#     if len(X) == 0:
#         print("No sequences found. Exiting.")
#         return

#     with open('lstm_scaler.pkl', 'rb') as f:
#         scaler = pickle.load(f)

#     flat_X = X.reshape(-1, X.shape[2])
#     scaled_flat = scaler.transform(flat_X)
#     X_scaled = scaled_flat.reshape(X.shape)

#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#     model = load_model('lstm_device_energy_model.h5')

#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)

#     save_metrics(y_train, y_train_pred, "LSTM_Train")
#     plot_pred_vs_actual(y_train, y_train_pred, "LSTM_Train")
#     plot_residuals(y_train, y_train_pred, "LSTM_Train")
#     plot_error_dist(y_train, y_train_pred, "LSTM_Train")

#     save_metrics(y_test, y_test_pred, "LSTM_Test")
#     plot_pred_vs_actual(y_test, y_test_pred, "LSTM_Test")
#     plot_residuals(y_test, y_test_pred, "LSTM_Test")
#     plot_error_dist(y_test, y_test_pred, "LSTM_Test")

#     print(f"Evaluation complete. Results saved to: {OUTPUT_FOLDER}/")

# if __name__ == "__main__":
#     evaluate_lstm_model()


#----

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Configuration
OUTPUT_FOLDER = 'evaluate_lstm_model'
TARGET_KEYS = ['powerConsumption', 'temperature', 'temperature_duration']
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

def load_enhanced_sequence_data(path='enhanced_synthetic_data.json', seq_length=3):
    with open(path, 'r') as f:
        raw_data = json.load(f)

    sequences, labels = [], []

    for device_id, records in raw_data.items():
        df = pd.DataFrame(records)
        if len(df) < seq_length + 1:
            continue
        df = df.sort_values('timestamp').fillna(-999)

        feature_data = df[FEATURE_KEYS].values
        target_data = df[TARGET_KEYS].values

        for i in range(len(df) - seq_length):
            seq = feature_data[i:i + seq_length]
            label = target_data[i + seq_length]
            sequences.append(seq)
            labels.append(label)

    return np.array(sequences), np.array(labels)

def plot_with_metrics(y_true, y_pred, key, output_subdir, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    metrics_text = f"MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.2f}"

    # 1. Prediction vs Actual
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name} - {key} - Predicted vs Actual\n{metrics_text}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, f"{key}_pred_vs_actual.png"))
    plt.close()

    # 2. Residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 5))
    plt.scatter(range(len(residuals)), residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Sample Index")
    plt.ylabel("Residual")
    plt.title(f"{name} - {key} - Residuals\n{metrics_text}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, f"{key}_residuals.png"))
    plt.close()

    # 3. Error Distribution
    errors = y_true - y_pred
    plt.figure(figsize=(6, 5))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.title(f"{name} - {key} - Error Distribution\n{metrics_text}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, f"{key}_error_distribution.png"))
    plt.close()

    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def make_combined_dashboard(metrics_dict, name):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, key in enumerate(TARGET_KEYS):
        m = metrics_dict[key]
        axs[i].bar(["MAE", "RMSE", "R²"], [m['mae'], m['rmse'], m['r2']], color="skyblue")
        axs[i].set_title(f"{key} - {name}")
        axs[i].grid(True)
    plt.suptitle(f"{name} - Combined Metrics Dashboard", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{name}_metrics_dashboard.png"))
    plt.close()

def evaluate_lstm_model():
    print("Loading sequences...")
    X, y = load_enhanced_sequence_data(seq_length=3)
    if len(X) == 0:
        print("No sequences found. Exiting.")
        return

    print("Scaling data...")
    with open('lstm_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    flat_X = X.reshape(-1, X.shape[2])
    X_scaled = scaler.transform(flat_X).reshape(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    print("Loading model...")
    model = load_model('lstm_device_energy_model.h5')

    print("Predicting...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("Generating plots...")
    for name, y_true, y_pred in [("LSTM_Train", y_train, y_train_pred),
                                 ("LSTM_Test", y_test, y_test_pred)]:
        subdir = os.path.join(OUTPUT_FOLDER, name)
        os.makedirs(subdir, exist_ok=True)
        metrics_summary = {}
        for i, key in enumerate(TARGET_KEYS):
            metrics = plot_with_metrics(
                y_true[:, i], y_pred[:, i], key, subdir, name)
            metrics_summary[key] = metrics
        make_combined_dashboard(metrics_summary, name)

    print(f"Evaluation complete. Results saved to: {OUTPUT_FOLDER}/")

if __name__ == '__main__':
    evaluate_lstm_model()