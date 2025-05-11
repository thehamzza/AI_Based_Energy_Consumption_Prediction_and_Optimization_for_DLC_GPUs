import numpy as np

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

def prepare_input_sequence(input_sequence, scaler):
    """
    Prepare a sequence of input records for LSTM inference.
    """
    feature_matrix = []
    for entry in input_sequence:
        row = [entry.get(key, -999) for key in FEATURE_KEYS]
        feature_matrix.append(row)

    feature_matrix = np.array(feature_matrix)
    scaled = scaler.transform(feature_matrix)
    return scaled.reshape(1, len(input_sequence), len(FEATURE_KEYS))

def predict_outputs(model, input_data):
    """
    Predict power consumption, temperature, and duration from model output.
    """
    prediction = model.predict(input_data)
    return {
        'predicted_power': prediction[0][0],
        'predicted_temperature': prediction[0][1],
        'predicted_duration': prediction[0][2]
    }