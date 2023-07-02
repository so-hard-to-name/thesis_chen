from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import xgboost as xgb
import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('data12h.csv')

data['sex'] = np.where(data['gender'] == "M", 0, 1)
data['heart_rate'] = np.where(abs(data['heart_rate_min'] - 80) >= abs(data['heart_rate_max'] - 80), data['heart_rate_min'], data['heart_rate_max'])
data['sbp'] = np.where(abs(data['sbp_min'] - 120) >= abs(data['sbp_max'] - 120), data['sbp_min'], data['sbp_max'])
data['dbp'] = np.where(abs(data['dbp_min'] - 80) >= abs(data['dbp_max'] - 80), data['dbp_min'], data['dbp_max'])
data['mbp'] = np.where(abs(data['mbp_min'] - 100) >= abs(data['mbp_max'] - 100), data['mbp_min'], data['mbp_max'])
data['resp_rate'] = np.where(abs(data['resp_rate_min'] - 14) >= abs(data['resp_rate_max'] - 14), data['resp_rate_min'], data['resp_rate_max'])
data['temperature'] = np.where(abs(data['temperature_min'] - 36) >= abs(data['temperature_max'] - 36), data['temperature_min'], data['temperature_max'])
data['spo2'] = np.where(abs(data['spo2_min'] - 98) >= abs(data['spo2_max'] - 98), data['spo2_min'], data['spo2_max'])

# Select the features and target variable
# features = data[['age','heart_rate_min', 'heart_rate_max', 'sbp_min', 'sbp_max', 'dbp_min', 'dbp_max', 'mbp_min', 'mbp_max', 'resp_rate_min', 'resp_rate_max' , 'temperature_min', 'temperature_max', 'spo2_min', 'spo2_max', 'urineoutput', 'gcs_value', 'gcs_motor', 'gcs_eyes', 'ventilation_code']]
features = data[['sex','age','heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate' , 'temperature', 'spo2', 'urineoutput', 'gcs_value', 'gcs_motor', 'ventilation_code']]
target = data['lods']

correlation_matrix = features.corr()

# Print the correlation matrix
print(correlation_matrix)

correlations = []
for feature in features.columns:
    correlation, _ = pearsonr(features[feature], target)
    correlations.append((feature, abs(correlation)))

# Sort the features based on their correlation strength
correlations.sort(key=lambda x: x[1], reverse=True)

print("Feature\t\t\tCorrelation")
for feature, correlation in correlations:
    print(f"{feature}\t\t{correlation:.8f}")

# Extract the feature names in the ranked order
ranked_features = [feat for feat, _ in correlations]

# Select the top features for training
top_features = ranked_features[:8]  # Adjust the number of top features as needed

# Subset the data with the top features
X_selected = features[top_features]

# Create an XGBoost model
xgboost_model = xgb.XGBRegressor(n_estimators=100, max_depth=3)

# Define the number of folds for cross-validation
k = 10
cv = KFold(n_splits=k, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(xgboost_model, X_selected, target, cv=cv, scoring='neg_mean_absolute_error')

# Convert the scores to positive values
scores = -scores

# Print the cross-validation scores
print("Cross-Validation MAE scores:", scores)
print("Average MAE:", scores.mean())
