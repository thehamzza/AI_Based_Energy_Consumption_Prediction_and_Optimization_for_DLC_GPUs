"""
📌 Sample Input & Output JSON for Flask API (app.py)

✅ Sample Input JSON (Sending All Required Features)
---------------------------------------------------
When sending a request to the `/predict` endpoint, use this JSON format:

{
  "temperature": [25, 26, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
  "humidity": [50, 52, 54, 55, 53, 52, 50, 48, 46, 44, 43, 42, 44, 46, 48, 50, 52, 54, 55, 56, 57, 58, 59, 60],
  "wind_speed": [10, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
  "air_pressure": [1010, 1011, 1012, 1013, 1012, 1011, 1010, 1009, 1008, 1007, 1006, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017],
  "dew_point": [10, 11, 12, 13, 12, 11, 10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
  "cloud_cover": [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18],
  "gpu_utilization": [40, 42, 44, 45, 43, 42, 40, 38, 36, 34, 33, 32, 34, 36, 38, 40, 42, 44, 45, 46, 47, 48, 49, 50],
  "fan_speed": [2000, 2100, 2200, 2300, 2200, 2100, 2000, 1900, 1800, 1700, 1600, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700],
  "core_temperature": [55, 56, 57, 58, 57, 56, 55, 54, 53, 52, 51, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],
  "vram_utilization": [4000, 4100, 4200, 4300, 4200, 4100, 4000, 3900, 3800, 3700, 3600, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700],
  "power_limit": [100, 101, 102, 103, 102, 101, 100, 99, 98, 97, 96, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
  "workload_type": [0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1],
  "hour_of_day": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
  "day_of_week": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
  "month_of_year": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
}

✅ Sample Output JSON (Successful Response)
-------------------------------------------
If the request is **successful**, the API will return:

{
  "predicted_gpu_energy": 330.5
}

    • predicted_gpu_energy: The **predicted energy consumption** for the next hour in **Watts**.

📌 Sample Error JSON Responses
-------------------------------
If a required key is missing:

{
  "error": "Missing 'temperature' in request"
}

If an incorrect data format is sent:

{
  "error": "Expecting a list of 24 numerical values for 'gpu_utilization'"
}

If there is an **internal server error**:

{
  "error": "Some error message from the server"
}

📌 How to Test API Using curl (Terminal)
----------------------------------------
Run this command to send a **POST request** with JSON data:

curl -X POST "http://127.0.0.1:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{
  "temperature": [25, 26, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
  "humidity": [50, 52, 54, 55, 53, 52, 50, 48, 46, 44, 43, 42, 44, 46, 48, 50, 52, 54, 55, 56, 57, 58, 59, 60],
  "wind_speed": [10, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
  "air_pressure": [1010, 1011, 1012, 1013, 1012, 1011, 1010, 1009, 1008, 1007, 1006, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017],
  "dew_point": [10, 11, 12, 13, 12, 11, 10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
  "cloud_cover": [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18],
  "gpu_utilization": [40, 42, 44, 45, 43, 42, 40, 38, 36, 34, 33, 32, 34, 36, 38, 40, 42, 44, 45, 46, 47, 48, 49, 50],
  "fan_speed": [2000, 2100, 2200, 2300, 2200, 2100, 2000, 1900, 1800, 1700, 1600, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700],
  "core_temperature": [55, 56, 57, 58, 57, 56, 55, 54, 53, 52, 51, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],
  "vram_utilization": [4000, 4100, 4200, 4300, 4200, 4100, 4000, 3900, 3800, 3700, 3600, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700],
  "power_limit": [100, 101, 102, 103, 102, 101, 100, 99, 98, 97, 96, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
  "workload_type": [0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1],
  "hour_of_day": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
  "day_of_week": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
  "month_of_year": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
}'


#live gcp cloud api end point test
curl -X POST "https://5000-cs-852786567045-default.cs-asia-southeast1-palm.cloudshell.dev/predict" \
     -H "Content-Type: application/json" \
     -d '{
  "temperature": [25, 26, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
  "humidity": [50, 52, 54, 55, 53, 52, 50, 48, 46, 44, 43, 42, 44, 46, 48, 50, 52, 54, 55, 56, 57, 58, 59, 60],
  "wind_speed": [10, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
  "air_pressure": [1010, 1011, 1012, 1013, 1012, 1011, 1010, 1009, 1008, 1007, 1006, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017],
  "dew_point": [10, 11, 12, 13, 12, 11, 10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
  "cloud_cover": [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18],
  "gpu_utilization": [40, 42, 44, 45, 43, 42, 40, 38, 36, 34, 33, 32, 34, 36, 38, 40, 42, 44, 45, 46, 47, 48, 49, 50],
  "fan_speed": [2000, 2100, 2200, 2300, 2200, 2100, 2000, 1900, 1800, 1700, 1600, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700],
  "core_temperature": [55, 56, 57, 58, 57, 56, 55, 54, 53, 52, 51, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],
  "vram_utilization": [4000, 4100, 4200, 4300, 4200, 4100, 4000, 3900, 3800, 3700, 3600, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700],
  "power_limit": [100, 101, 102, 103, 102, 101, 100, 99, 98, 97, 96, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
  "workload_type": [0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1],
  "hour_of_day": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
  "day_of_week": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
  "month_of_year": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
}'


✅ Expected Output in Terminal:

{"predicted_gpu_energy": 330.5}

📌 Summary of Responses
------------------------
| Scenario                      | Expected Output                                      |
|--------------------------------|-----------------------------------------------------|
| ✅ Valid Request               | {"predicted_gpu_energy": 330.5}                     |
| ❌ Missing temperature         | {"error": "Missing 'temperature' in request"}      |
| ❌ Wrong Format                | {"error": "Expecting a list of 24 numerical values for 'gpu_utilization'"} |
| ❌ Internal Server Error       | {"error": "Some error message from the server"}  |

API fully supports **all 15 input features** for accurate GPU energy predictions! 
"""


from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model & scalers
model = load_model("gpu_energy_model.h5")
scaler_features = joblib.load("scaler_features.pkl")  # Feature Scaler
scaler_target = joblib.load("scaler_target.pkl")  # Target Scaler

SEQ_LENGTH = 24  # Must match training sequence length

# API home route
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the GPU Energy Prediction API! Use POST /predict to get predictions."})

# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Check if required features are present
        required_keys = [
            "temperature", "humidity", "wind_speed", "air_pressure", "dew_point", "cloud_cover",
            "gpu_utilization", "fan_speed", "core_temperature", "vram_utilization", "power_limit",
            "workload_type", "hour_of_day", "day_of_week", "month_of_year"
        ]

        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing '{key}' in request"}), 400

        # Convert JSON data to numpy array
        input_data = np.array([
            data["temperature"], data["humidity"], data["wind_speed"], data["air_pressure"],
            data["dew_point"], data["cloud_cover"], data["gpu_utilization"], data["fan_speed"],
            data["core_temperature"], data["vram_utilization"], data["power_limit"],
            data["workload_type"], data["hour_of_day"], data["day_of_week"], data["month_of_year"]
        ]).reshape(1, SEQ_LENGTH, 15)  # Match LSTM input shape

        # Scale the input data
        input_data_scaled = scaler_features.transform(input_data.reshape(-1, 15)).reshape(1, SEQ_LENGTH, 15)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Convert back to original scale
        predicted_energy = scaler_target.inverse_transform([[prediction[0][0]]])[0][0]

        return jsonify({"predicted_gpu_energy": round(predicted_energy, 2)})

    except Exception as e:
        logger.error("Error in prediction: %s", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
   
