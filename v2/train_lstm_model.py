import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Final and confirmed list of features available in your dataset
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

def load_and_prepare_sequence(filepath, seq_length=3):
    with open(filepath, 'r') as f:
        device_data = json.load(f)

    all_sequences = []
    all_targets = []

    for device_id, records in device_data.items():
        df = pd.DataFrame(records)

        if len(df) < seq_length + 1:
            continue

        df.sort_values('timestamp', inplace=True)

        # Select features and fill missing
        feature_data = df[FEATURE_KEYS].fillna(-999).values
        target_data = df[['powerConsumption', 'temperature', 'temperature_duration']].values

        for i in range(len(df) - seq_length):
            sequence = feature_data[i:i + seq_length]
            target = target_data[i + seq_length]
            all_sequences.append(sequence)
            all_targets.append(target)

    if not all_sequences:
        return np.array([]), np.array([]), None

    X = np.array(all_sequences)
    y = np.array(all_targets)

    # Scale inputs
    flat_X = X.reshape(-1, X.shape[2])
    scaler = StandardScaler().fit(flat_X)
    X_scaled = scaler.transform(flat_X).reshape(X.shape)

    return X_scaled, y, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3))  # Predicting 3 values
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_lstm():
    print("Loading and preparing sequence data...")
    X, y, scaler = load_and_prepare_sequence('enhanced_synthetic_data.json', seq_length=3)

    if len(X) == 0:
        print("No sequences found. Aborting training.")
        return

    print(f"Input shape: {X.shape}, Output shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building LSTM model...")
    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))

    print("Training model...")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Evaluation complete. Test MAE: {mae:.2f}")

    print("Saving model and scaler...")
    model.save('lstm_device_energy_model.h5')
    with open('lstm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Training complete. Model and scaler saved.")

if __name__ == '__main__':
    train_lstm()