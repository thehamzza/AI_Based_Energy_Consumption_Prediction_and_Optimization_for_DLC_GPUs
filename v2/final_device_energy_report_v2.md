# 📘 Complete Technical Documentation – Coolth Store V2 Energy Prediction System

This document provides a **comprehensive breakdown** of the V2 implementation, its structure, functionality, analysis, and visual outputs.

## 📁 Directory Structure & File Purposes

```text
project-root/
│
├── app.py                             # Flask API application
├── evaluate_lstm_model.py            # Model evaluation and visualization
├── feature_engineering.py            # Feature generation and transformation
├── train_lstm_model.py               # LSTM model training
├── shap_lstm_explain.py              # SHAP explanation for LSTM
├── feature_analysis.py               # Correlation + Permutation importance
├── generate_gpu_data.py              # Synthetic GPU data generator
├── generate_refrigerator_data.py     # Synthetic refrigerator data generator
├── infer_local.py                    # Manual inference testing script
├── utils.py                          # Utility functions for preprocessing
├── requirements.txt                  # Dependency list
├── Dockerfile                        # For containerized deployment on Cloud Run
├── *.json, *.pkl, *.h5               # Data, model weights, and scalers
├── feature_analysis_results/         # SHAP, correlation & importance images
├── evaluate_lstm_model/              # Evaluation plots (actual vs predicted)
└── GUI AI DLC.png                    # UI/UX preview of dashboard
```

Each Python script, model file, and image plays a specific role in the **end-to-end AI pipeline** for device energy prediction.

## ⚙️ Feature Engineering

- **Script**: `feature_engineering.py`

- **Output**: `enhanced_synthetic_data.json`

This script transforms raw data into feature-rich sequences including:

- Rolling mean & std over 3 hours
- Previous hour metrics
- Difference features (diff)
- Time features: hour, day, weekend/night

📸 **Figure**: `feature engineering.png`
This image visualizes the complete feature transformation pipeline from raw device logs to model-ready inputs.

## 🧠 Model Architecture – LSTM

The LSTM model (`train_lstm_model.py`) is trained with a 3-step time window to predict:
- Power Consumption
- Core Temperature
- Duration of elevated temperature

- **Model File**: `lstm_device_energy_model.h5`
- **Scaler**: `lstm_scaler.pkl`

## 📊 Evaluation Outputs

Script: `evaluate_lstm_model.py` generates 3 kinds of plots for each output:
- **Predicted vs Actual**
- **Residuals**
- **Error Distribution**


Each plot embeds evaluation metrics (MAE, RMSE, R²) for clarity.

📸 **Evaluation Plots** (Located in `evaluate_lstm_model/`):

- LSTM_Test_powerConsumption_pred_vs_actual.png
- LSTM_Test_temperature_residuals.png
- LSTM_Test_temperature_duration_error_distribution.png

These images **visually assess model performance** and highlight potential bias, variance, or underfitting/overfitting.

## 🔬 Feature Importance Analysis

Script: `feature_analysis.py` uses:
- **Correlation Heatmap**
- **Permutation Importance** (Random Forest)

📸 **Images:**

- `correlation_heatmap.png`: Visualizes linear dependencies between features.

- `permutation_importance.png`: Quantifies how much model performance drops if a feature is randomly permuted.

## 💡 SHAP Interpretability – LSTM Model

Script: `shap_lstm_explain.py` produces SHAP explanations for each target variable.
📸 **Images:**

- `shap_lstm_power.png` → Importance of features on power consumption
- `shap_lstm_temperature.png` → On temperature prediction
- `shap_lstm_duration.png` → On temperature duration

## 🌐 API – Live Endpoint Details

- **Live Endpoint:** [Device Energy API V2](https://device-energy-api-v2-255530078026.us-central1.run.app/)

- **Prediction Endpoint:** `/predict`

- **Method**: `POST`

### 🔁 Input JSON Format:
```json
[
  {
    "ambientTemperature": 21.5,
    "powerConsumption": 110.0,
    "workLoadType": 1,
    ... (20+ features)
  },
  { ... },
  { ... }
]
```

### ✅ Example Output:
```json
{
  "core_temperature": 62.68,
  "predicted_power_consumption": 97.55,
  "temperature_duration": 2.36
}
```

## 🖥 Dashboard Preview
📸 **Image:** `GUI AI DLC.png`
This dashboard visual presents model predictions and user uploads, suitable for internal monitoring or client reporting.

## 📌 Final Observations
- V2 is deployed as an isolated Cloud Run container
- Modular structure enables plug-and-play for new models or features
- Each component is tested for compatibility
