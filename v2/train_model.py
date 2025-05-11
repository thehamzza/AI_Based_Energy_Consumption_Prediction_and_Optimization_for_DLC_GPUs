import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def load_all_synthetic_data():
    """
    Loads and combines all synthetic data from JSON files ending with '_synthetic_data.json' in the current directory.

    Returns:
        list: A combined list of all records from the matched JSON files.
    """
    data = []
    for file in os.listdir():
        if file.endswith('_synthetic_data.json'):
            try:
                with open(file, 'r') as f:
                    records = json.load(f)
                    data.extend(records)
                    print(f"Loaded {len(records)} records from {file}")
            except json.JSONDecodeError as e:
                print(f"Error reading {file}: {e}")
    if not data:
        print("No synthetic data files found.")
    return data

def preprocess_data(data):
    """
    Converts raw list of device records to a DataFrame and extracts features and target.

    Args:
        data (list): List of device records as dictionaries.

    Returns:
        X (ndarray): Scaled feature array.
        y (ndarray): Target array (power consumption).
        scaler (StandardScaler): Fitted scaler object for saving.
    """
    df = pd.DataFrame(data)

    # Drop non-numeric or identifier columns
    drop_cols = ['uidOfDevice', 'typeOfDevice', 'typeDescription', 'timeStamp', 'doorOpen', 'doorClose']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Separate features and target
    X = df.drop(columns=['powerConsumption'])
    y = df['powerConsumption']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, scaler

def build_model(input_dim):
    """
    Builds and compiles a simple feedforward neural network model.

    Args:
        input_dim (int): Number of input features.

    Returns:
        keras.Model: Compiled model ready for training.
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer for regression

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(model_output_path, scaler_output_path):
    """
    Loads and merges all available synthetic device data, preprocesses it, trains a model, and saves it.

    Args:
        model_output_path (str): File path to save the trained model (e.g., 'device_energy_model.h5').
        scaler_output_path (str): File path to save the fitted feature scaler (e.g., 'scaler_features.pkl').
    """
    # Load and prepare data
    data = load_all_synthetic_data()
    if not data:
        print("No data available for training.")
        return

    X, y, scaler = preprocess_data(data)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(X.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=16, callbacks=[early_stop], verbose=1)

    # Evaluate on test set
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Mean Absolute Error: {mae:.2f}")

    # Save model and scaler
    model.save(model_output_path)
    with open(scaler_output_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Model saved to {model_output_path}")
    print(f"Scaler saved to {scaler_output_path}")

if __name__ == "__main__":
    # You can change the output paths as needed
    train_model('device_energy_model.h5', 'scaler_features.pkl')