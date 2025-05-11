
# 🔧 Device Energy Prediction API — Technical Report

## Overview
This API predicts three outputs for a given device using historical and contextual data:

- **Predicted Power Consumption** (in watts)
- **Core Temperature** (in degrees Celsius)
- **Temperature Duration** (in hours)

It uses an LSTM-based model trained on time-sequenced data for high temporal accuracy.

---

## 📍 API Endpoint

### **POST /predict**

**Purpose:** Predict the next time step’s device behavior based on the last 3 records of input.

---

## 📥 Input JSON

### Format:
```json
{
  "sequence": [
    {
      "ambientTemperature": 21.5,
      "powerConsumption": 110.0,
      "workLoadType": 1,
      "hour_of_day": 11,
      "day_of_week": 2,
      "month_of_year": 4,
      "device_on": 1,
      "open_duration": 1.2,
      "power_prev_hour": 100.0,
      "temp_prev_hour": 20.0,
      "duration_prev_hour": 1.0,
      "rolling_power_mean_3h": 105.0,
      "rolling_temp_mean_3h": 20.5,
      "rolling_duration_mean_3h": 1.1,
      "rolling_power_std_3h": 2.5,
      "rolling_temp_std_3h": 0.6,
      "rolling_duration_std_3h": 0.3,
      "power_diff": 3.0,
      "temp_diff": 0.5,
      "duration_diff": 0.2,
      "is_weekend": 0,
      "is_night": 0
    },
    { /* 2nd record */ },
    { /* 3rd record */ }
  ]
}
```

- **Minimum Length**: 3 entries (sliding window input for LSTM).
- **Field Order**: Not important in JSON, but all fields must be present per record.

---

## 📤 Output JSON

### Format:
```json
{
  "predicted_power_consumption": 108.1,
  "core_temperature": 68.23,
  "temperature_duration": 1.45
}
```

- Output values are **floats**.
- These represent the next predicted hour’s values for power, core temperature, and duration.

---

## 🔄 Units Summary

| Field                      | Unit        | Description                                  |
|---------------------------|-------------|----------------------------------------------|
| ambientTemperature         | °C          | Ambient temperature at the time              |
| powerConsumption           | Watts       | Device's actual power use                    |
| open_duration              | Hours       | Duration the device was in active/open state |
| power_prev_hour            | Watts       | Previous hour's power use                    |
| temp_prev_hour             | °C          | Previous hour's core temp                    |
| duration_prev_hour         | Hours       | Duration for previous reading                |
| rolling_*_mean_3h          | Same as var | 3-hour rolling average                       |
| rolling_*_std_3h           | Same as var | 3-hour rolling std. deviation                |
| *_diff                     | Same as var | Difference between recent time steps         |
| is_night, is_weekend       | Binary      | 0 or 1 flags for time context                |

---

## ✅ Example Request

**POST** `/predict`

```json
{
  "sequence": [
    {
      "ambientTemperature": 21.5,
      "powerConsumption": 110.0,
      "workLoadType": 1,
      "hour_of_day": 11,
      "day_of_week": 2,
      "month_of_year": 4,
      "device_on": 1,
      "open_duration": 1.2,
      "power_prev_hour": 100.0,
      "temp_prev_hour": 20.0,
      "duration_prev_hour": 1.0,
      "rolling_power_mean_3h": 105.0,
      "rolling_temp_mean_3h": 20.5,
      "rolling_duration_mean_3h": 1.1,
      "rolling_power_std_3h": 2.5,
      "rolling_temp_std_3h": 0.6,
      "rolling_duration_std_3h": 0.3,
      "power_diff": 3.0,
      "temp_diff": 0.5,
      "duration_diff": 0.2,
      "is_weekend": 0,
      "is_night": 0
    },
    { /* 2nd record */ },
    { /* 3rd record */ }
  ]
}
```

---

## ✅ Example Response

```json
{
  "predicted_power_consumption": 97.58,
  "core_temperature": 62.70,
  "temperature_duration": 2.36
}
```

---

## 🛠 Backend Stack

- **Framework**: Flask (Python)
- **Model**: LSTM (TensorFlow Keras)
- **Scaler**: Scikit-learn (MinMaxScaler or StandardScaler)
- **Deployment**: Exposed on `http://<host>:5001/predict`

---

## 🧪 Logging and Debugging

- Logs request payloads and model input shape
- Catches and reports NaN or invalid prediction values
- Use Postman or CURL for manual endpoint testing

---

## ⚠️ Notes

- All fields are mandatory per timestep in `sequence`
- JSON key order does **not** matter, but keys must be present
- Returns HTTP `500` on model/scaling errors or malformed inputs
