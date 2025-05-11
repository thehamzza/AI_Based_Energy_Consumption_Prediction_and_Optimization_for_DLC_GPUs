import json
import joblib
from tensorflow.keras.models import load_model
from utils import prepare_input_sequence, predict_outputs

def load_model_and_scaler(model_path, scaler_path):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def main():
    with open('enhanced_synthetic_data.json', 'r') as f:
        data = json.load(f)

    if not isinstance(data, dict) or not data:
        print("No device data found in enhanced_synthetic_data.json")
        return

    # Dynamically select the first device
    device_id = list(data.keys())[0]
    sequence = data[device_id]

    if len(sequence) < 3:
        print(f"Not enough data for device {device_id} to form a sequence.")
        return

    input_sequence = sequence[-3:]

    model, scaler = load_model_and_scaler('lstm_device_energy_model.h5', 'lstm_scaler.pkl')
    input_seq = prepare_input_sequence(input_sequence, scaler)

    predictions = predict_outputs(model, input_seq)

    print(f"Device ID: {device_id}")
    print(f"Predicted Power Consumption: {round(predictions['predicted_power'], 2)} Watts")
    print(f"Predicted Temperature: {round(predictions['predicted_temperature'], 2)} °C")
    print(f"Predicted Duration: {round(predictions['predicted_duration'], 2)} minutes")

if __name__ == '__main__':
    main()