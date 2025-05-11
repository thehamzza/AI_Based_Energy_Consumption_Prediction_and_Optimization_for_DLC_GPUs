# from flask import Flask, request, jsonify
# import joblib
# import json
# import numpy as np
# import logging
# from tensorflow.keras.models import load_model
# from utils import prepare_input_sequence, predict_outputs

# app = Flask(__name__)

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

# # Setup logging
# logging.basicConfig(level=logging.DEBUG)

# def load_model_and_scaler(model_path, scaler_path):
#     try:
#         model = load_model(model_path)
#         scaler = joblib.load(scaler_path)
#         return model, scaler
#     except Exception as e:
#         logging.error(f"Error loading model or scaler: {e}")
#         raise e

# @app.route('/')
# def home():
#     return jsonify({"message": "Device Energy Prediction API is running."})

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         content = request.get_json()
#         if not content or 'sequence' not in content:
#             return jsonify({"error": "Missing 'sequence' in request"}), 400

#         input_sequence = content['sequence']

#         if len(input_sequence) < 3:
#             return jsonify({"error": "Input sequence must be at least 3 records long"}), 400

#         # Load model and scaler
#         model, scaler = load_model_and_scaler('lstm_device_energy_model.h5', 'lstm_scaler.pkl')

#         # Prepare input
#         logging.debug(f"Preparing input sequence: {input_sequence[-3:]}")
#         input_seq = prepare_input_sequence(input_sequence[-3:], scaler)
#         logging.debug(f"Input sequence after preparation: {input_seq}")

#         input_seq = np.array(input_seq)
#         logging.debug(f"Input sequence shape: {input_seq.shape}")

#         # Predict
#         prediction = predict_outputs(model, input_seq)
#         logging.debug(f"Raw prediction: {prediction}")

#         required_keys = ['predicted_power', 'predicted_temperature', 'predicted_duration']
#         if not isinstance(prediction, dict) or not all(k in prediction for k in required_keys):
#             raise ValueError("Prediction dictionary format is invalid.")
#         if any(np.isnan(prediction[k]) or np.isinf(prediction[k]) for k in required_keys):
#             raise ValueError(f"Prediction contains invalid values: {prediction}")

#         return jsonify({
#             "predicted_power_consumption": float(prediction["predicted_power"]),
#             "core_temperature": float(prediction["predicted_temperature"]),
#             "temperature_duration": float(prediction["predicted_duration"])
#         })

#     except Exception as e:
#         logging.error(f"Error during prediction: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5001, debug=True)


from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
from tensorflow.keras.models import load_model
from utils import prepare_input_sequence, predict_outputs

app = Flask(__name__)

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

# Setup logging
logging.basicConfig(level=logging.DEBUG)

def load_model_and_scaler(model_path, scaler_path):
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading model or scaler: {e}")
        raise e

@app.route('/')
def home():
    return jsonify({"message": "Device Energy Prediction API is running."})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.get_json()

        if not isinstance(content, list) or len(content) < 3:
            return jsonify({"error": "Input must be a list of at least 3 records."}), 400

        input_sequence = content[-3:]  # take the last 3

        model, scaler = load_model_and_scaler('lstm_device_energy_model.h5', 'lstm_scaler.pkl')

        logging.debug(f"Preparing input sequence: {input_sequence}")
        input_seq = prepare_input_sequence(input_sequence, scaler)
        input_seq = np.array(input_seq)
        logging.debug(f"Input sequence shape: {input_seq.shape}")

        prediction = predict_outputs(model, input_seq)
        logging.debug(f"Raw prediction: {prediction}")

        required_keys = ['predicted_power', 'predicted_temperature', 'predicted_duration']
        if not isinstance(prediction, dict) or not all(k in prediction for k in required_keys):
            raise ValueError("Prediction dictionary format is invalid.")
        if any(np.isnan(prediction[k]) or np.isinf(prediction[k]) for k in required_keys):
            raise ValueError(f"Prediction contains invalid values: {prediction}")

        return jsonify({
            "predicted_power_consumption": float(prediction["predicted_power"]),
            "core_temperature": float(prediction["predicted_temperature"]),
            "temperature_duration": float(prediction["predicted_duration"])
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)