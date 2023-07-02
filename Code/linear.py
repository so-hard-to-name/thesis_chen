import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset into a pandas DataFrame
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
features = data[['sex','age','heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate' , 'temperature', 'spo2', 'urineoutput', 'gcs_value', 'gcs_motor', 'gcs_eyes', 'ventilation_code']]
target = data['lods']

# Calculate Spearman correlation for each feature
correlations = []
for feature in features.columns:
    corr, _ = spearmanr(features[feature], target)
    correlations.append((feature, abs(corr)))

correlations.sort(key=lambda x: x[1], reverse=True)

# Print Spearman correlation coefficients
print("Spearman Correlation Coefficients:")
for feature, corr in correlations:
    print(f"{feature}\t\t{corr:.8f}")

# correlations, _ = spearmanr(features, target)

# # Print Spearman correlation coefficients
# print("Spearman Correlation Coefficients:")
# for i, feature in enumerate(features.columns):
#     print(f"{feature}: {correlations[i]}")

# Extract the feature names in the ranked order
ranked_features = [feat for feat, _ in correlations]

# Select the top features for training
top_features = ranked_features[:14]  # Adjust the number of top features as needed

# Subset the data with the top features
X_selected = features[top_features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, target, test_size=0.2, random_state=42)

# Create a linear regression object
regressor = LinearRegression()

# Train the model using the training data
regressor.fit(X_train, y_train)

# Predict on the test data
y_pred = regressor.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('Root Mean Squared Error:', rmse)
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

    # Predicted vs Actural
plt.scatter(y_pred, y_test)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Actual Plot')
plt.show()
