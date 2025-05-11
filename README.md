# 📘 Complete Technical Documentation – Coolth Store V2 Energy Prediction System

This document provides a **comprehensive breakdown** of the V2 implementation, its structure, functionality, analysis, and visual outputs.

## 📁 Directory Structure & File Purposes

```text
project-root/
│
├── app.py                             # Flask API (main endpoint)
├── device_energy_prediction_api.md    # API documentation (markdown)
├── Dockerfile                         # For Cloud Run deployment
├── evaluate_lstm_model.py             # Evaluates model and plots results
├── feature_engineering.py             # Adds rolling stats, lag features etc.
├── generate_gpu_data.py               # Generates synthetic GPU JSON data
├── generate_refrigerator_data.py      # Generates synthetic refrigerator data
├── gpu_energy_model.h5                # Pretrained GPU model (v1)
├── gpu_synthetic_data.json            # Synthetic data (GPU type)
├── GUI AI DLC.png                     # Mockup of front-end dashboard
├── infer_local.py                     # Run inference locally for testing
├── lstm_device_energy_model.h5        # Final trained LSTM model weights
├── lstm_scaler.pkl                    # Feature scaler (used in app.py)
├── refrigerator_synthetic_data.json   # Synthetic data (Refrigerator type)
├── requirements.txt                   # All required Python libraries
├── scaler_features.pkl                # Backup of scaler for features
├── shap_lstm_explain.py               # SHAP explanations for LSTM predictions
├── train_lstm_model.py                # LSTM training script
├── train_model.py                     # Deprecated/general training logic
├── utils.py                           # Reusable helper functions

evaluate_lstm_model/
├── LSTM_Test/
    ├── LSTM_Test_powerConsumption_pred_vs_actual.png
    ├── LSTM_Test_powerConsumption_residuals.png
    ├── LSTM_Test_powerConsumption_error_distribution.png
    ├── LSTM_Test_temperature_pred_vs_actual.png
    ├── LSTM_Test_temperature_residuals.png
    ├── LSTM_Test_temperature_error_distribution.png
    ├── LSTM_Test_temperature_duration_pred_vs_actual.png
    ├── LSTM_Test_temperature_duration_residuals.png
    ├── LSTM_Test_temperature_duration_error_distribution.png
    ├── LSTM_Test_metrics_dashboard.png
├── LSTM_Train/
    ├── LSTM_Train_powerConsumption_pred_vs_actual.png
    ├── LSTM_Train_powerConsumption_residuals.png
    ├── LSTM_Train_powerConsumption_error_distribution.png
    ├── LSTM_Train_temperature_pred_vs_actual.png
    ├── LSTM_Train_temperature_residuals.png
    ├── LSTM_Train_temperature_error_distribution.png
    ├── LSTM_Train_temperature_duration_pred_vs_actual.png
    ├── LSTM_Train_temperature_duration_residuals.png
    ├── LSTM_Train_temperature_duration_error_distribution.png
    ├── LSTM_Train_metrics_dashboard.png

├── feature_analysis_results/
    ├── correlation_heatmap.png
    ├── permutation_importance.png
    ├── shap_beeswarm_plot.png
    ├── shap_lstm_outputs/
        ├── shap_lstm_duration.png
        ├── shap_lstm_power.png
        ├── shap_lstm_temperature.png
│
└── v2/   # GitHub branch / working directory
    ├── All above core files listed above were synced from this directory
    └── Note: mirrors final state of project folder for v2 branch
```

Each Python script, model file, and image plays a specific role in the **end-to-end AI pipeline** for device energy prediction.

## 📊 Dashboard UI Preview

This image presents a conceptual design of the user-facing dashboard where real-time predictions can be visualized.

![GUI Dashboard](v2/GUI%20AI%20DLC.png)


## ⚙️ Feature Engineering

- **Script**: `feature_engineering.py`

- **Output**: `enhanced_synthetic_data.json`

This script transforms raw data into feature-rich sequences including:

- Rolling mean & std over 3 hours
- Previous hour metrics
- Difference features (diff)
- Time features: hour, day, weekend/night

![Feature Engineering Diagram](v2/feature%20engineering.png)

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

---
### 📊 Evaluation Dashboard — LSTM (Test Data)

Shows MAE, RMSE, and R² for power, temperature, and duration predictions on test data.

![LSTM Test Dashboard](v2/evaluate_lstm_model/LSTM_Test_metrics_dashboard.png)

---

### 📊 Evaluation Dashboard — LSTM (Train Data)

Similar metrics dashboard but for training data, useful for overfitting detection.

![LSTM Train Dashboard](v2/evaluate_lstm_model/LSTM_Train_metrics_dashboard.png)

---
# 📊 LSTM Evaluation Results – Detailed Visual Report

## 🧪 LSTM Test Set Evaluation

### 📉 Combined Metrics Dashboard
![LSTM Test Dashboard](evaluate_lstm_model/LSTM_Test_metrics_dashboard.png)

### 📍 LSTM Test powerConsumption pred vs actual
![LSTM Test powerConsumption pred vs actual](evaluate_lstm_model/LSTM_Test_powerConsumption_pred_vs_actual.png)

### 📍 LSTM Test powerConsumption residuals
![LSTM Test powerConsumption residuals](evaluate_lstm_model/LSTM_Test_powerConsumption_residuals.png)

### 📍 LSTM Test powerConsumption error distribution
![LSTM Test powerConsumption error distribution](evaluate_lstm_model/LSTM_Test_powerConsumption_error_distribution.png)

### 📍 LSTM Test temperature pred vs actual
![LSTM Test temperature pred vs actual](evaluate_lstm_model/LSTM_Test_temperature_pred_vs_actual.png)

### 📍 LSTM Test temperature residuals
![LSTM Test temperature residuals](evaluate_lstm_model/LSTM_Test_temperature_residuals.png)

### 📍 LSTM Test temperature error distribution
![LSTM Test temperature error distribution](evaluate_lstm_model/LSTM_Test_temperature_error_distribution.png)

### 📍 LSTM Test temperature duration pred vs actual
![LSTM Test temperature duration pred vs actual](evaluate_lstm_model/LSTM_Test_temperature_duration_pred_vs_actual.png)

### 📍 LSTM Test temperature duration residuals
![LSTM Test temperature duration residuals](evaluate_lstm_model/LSTM_Test_temperature_duration_residuals.png)

### 📍 LSTM Test temperature duration error distribution
![LSTM Test temperature duration error distribution](evaluate_lstm_model/LSTM_Test_temperature_duration_error_distribution.png)

## 🏋️‍♂️ LSTM Train Set Evaluation

### 📉 Combined Metrics Dashboard
![LSTM Train Dashboard](evaluate_lstm_model/LSTM_Train_metrics_dashboard.png)

### 📍 LSTM Train powerConsumption pred vs actual
![LSTM Train powerConsumption pred vs actual](evaluate_lstm_model/LSTM_Train_powerConsumption_pred_vs_actual.png)

### 📍 LSTM Train powerConsumption residuals
![LSTM Train powerConsumption residuals](evaluate_lstm_model/LSTM_Train_powerConsumption_residuals.png)

### 📍 LSTM Train powerConsumption error distribution
![LSTM Train powerConsumption error distribution](evaluate_lstm_model/LSTM_Train_powerConsumption_error_distribution.png)

### 📍 LSTM Train temperature pred vs actual
![LSTM Train temperature pred vs actual](evaluate_lstm_model/LSTM_Train_temperature_pred_vs_actual.png)

### 📍 LSTM Train temperature residuals
![LSTM Train temperature residuals](evaluate_lstm_model/LSTM_Train_temperature_residuals.png)

### 📍 LSTM Train temperature error distribution
![LSTM Train temperature error distribution](evaluate_lstm_model/LSTM_Train_temperature_error_distribution.png)

### 📍 LSTM Train temperature duration pred vs actual
![LSTM Train temperature duration pred vs actual](evaluate_lstm_model/LSTM_Train_temperature_duration_pred_vs_actual.png)

### 📍 LSTM Train temperature duration residuals
![LSTM Train temperature duration residuals](evaluate_lstm_model/LSTM_Train_temperature_duration_residuals.png)

### 📍 LSTM Train temperature duration error distribution
![LSTM Train temperature duration error distribution](evaluate_lstm_model/LSTM_Train_temperature_duration_error_distribution.png)


---

## 🔬 Feature Importance Analysis

Script: `feature_analysis.py` uses:
- **Correlation Heatmap**
- **Permutation Importance** (Random Forest)

📸 **Images:**

- ### 🔍 Correlation Heatmap

This heatmap displays Pearson correlation coefficients between features and targets. Darker shades indicate stronger correlations (positive or negative).

![Correlation Heatmap](v2/feature_analysis_results/correlation_heatmap.png)

- `permutation_importance.png`: Quantifies how much model performance drops if a feature is randomly permuted.

---

### 🔍 Permutation Feature Importance

This bar chart illustrates the relative importance of each input feature based on how shuffling its values affects model performance.

![Permutation Importance](v2/feature_analysis_results/permutation_importance.png)

---

### 🔍 SHAP Beeswarm Plots

#### 🧠 SHAP — Power Consumption
Highlights which features most influence the model's prediction of power usage. Each dot represents a prediction and color encodes feature value.

![SHAP Power](v2/feature_analysis_results/shap_lstm_outputs/shap_lstm_power.png)

#### 🧠 SHAP — Temperature
Visualizes feature impact on core temperature predictions. Strong contributors include ambient temperature and rolling averages.

![SHAP Temperature](v2/feature_analysis_results/shap_lstm_outputs/shap_lstm_temperature.png)

#### 🧠 SHAP — Duration
Analyzes how each feature influences predicted duration of elevated temperature in the device.

![SHAP Duration](v2/feature_analysis_results/shap_lstm_outputs/shap_lstm_duration.png)

---

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


## 📌 Final Observations
- V2 is deployed as an isolated Cloud Run container
- Modular structure enables plug-and-play for new models or features
- Each component is tested for compatibility
