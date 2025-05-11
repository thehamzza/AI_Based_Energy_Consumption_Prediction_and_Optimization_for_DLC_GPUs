"""
This script includes:
	1.	Correlation Heatmap
	2.	Permutation Feature Importance
	3.	SHAP Beeswarm Plot

How to use:

pip install shap seaborn scikit-learn matplotlib
python feature_analysis.py

Output Files Generated:
	•	correlation_heatmap.png
	•	permutation_importance.png
	•	shap_beeswarm_plot.png
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap

# Output directory for all visualizations
OUTPUT_DIR = "feature_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the JSON data
with open("enhanced_synthetic_data.json") as f:
    raw_data = json.load(f)

# Flatten if data is nested
if isinstance(raw_data, dict):
    records = []
    for device_data in raw_data.values():
        records.extend(device_data)
else:
    records = raw_data

df = pd.DataFrame(records)
print(f"Data loaded. Rows: {len(df)} Columns: {len(df.columns)}")

# Drop unwanted/non-numeric columns
drop_cols = ['device_id', 'timestamp', 'doorOpen', 'doorClose']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

# Filter numeric features
df = df.select_dtypes(include=['float64', 'int64'])

# Correlation Heatmap
if df.empty:
    print("No numeric features available. Exiting.")
    exit()

plt.figure(figsize=(18, 14))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, annot_kws={"size": 10})
plt.title("Correlation Heatmap", fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=300)
plt.close()
print("correlation_heatmap.png saved.")

# Train model for feature importance
target = "powerConsumption"
if target not in df.columns:
    print("Target column 'powerConsumption' not found. Exiting.")
    exit()

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Permutation Importance
perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
perm_importance = pd.Series(perm.importances_mean, index=X.columns)
perm_importance.sort_values().plot(kind='barh', figsize=(12, 8), title="Permutation Feature Importance")
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "permutation_importance.png"), dpi=300)
plt.close()
print("permutation_importance.png saved.")

# SHAP Beeswarm Plot
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:100])

plt.figure(figsize=(12, 8))
shap.plots.beeswarm(shap_values, show=False, max_display=25)
plt.title("SHAP Beeswarm Plot", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm_plot.png"), dpi=300)
plt.close()
print("shap_beeswarm_plot.png saved.")