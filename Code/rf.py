import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Read the CSV file into a DataFrame
data = pd.read_csv('data12h.csv')

data['sex'] = np.where(data['gender'] == "M", 0, 1)
data['heart_rate'] = np.where(abs(data['heart_rate_min'] - 65) >= abs(data['heart_rate_max'] - 65), data['heart_rate_min'], data['heart_rate_max'])
data['sbp'] = np.where(abs(data['sbp_min'] - 120) >= abs(data['sbp_max'] - 120), data['sbp_min'], data['sbp_max'])
data['dbp'] = np.where(abs(data['dbp_min'] - 80) >= abs(data['dbp_max'] - 80), data['dbp_min'], data['dbp_max'])
data['mbp'] = np.where(abs(data['mbp_min'] - 100) >= abs(data['mbp_max'] - 100), data['mbp_min'], data['mbp_max'])
data['resp_rate'] = np.where(abs(data['resp_rate_min'] - 14) >= abs(data['resp_rate_max'] - 14), data['resp_rate_min'], data['resp_rate_max'])
data['temperature'] = np.where(abs(data['temperature_min'] - 36) >= abs(data['temperature_max'] - 36), data['temperature_min'], data['temperature_max'])
data['spo2'] = np.where(abs(data['spo2_min'] - 98) >= abs(data['spo2_max'] - 98), data['spo2_min'], data['spo2_max'])

# Select the features and target variable
# features = data[['age','heart_rate_min', 'heart_rate_max', 'sbp_min', 'sbp_max', 'dbp_min', 'dbp_max', 'mbp_min', 'mbp_max', 'resp_rate_min', 'resp_rate_max' , 'temperature_min', 'temperature_max', 'spo2_min', 'spo2_max', 'urineoutput', 'gcs_value', 'gcs_motor', 'gcs_eyes', 'ventilation_code']]
features = data[['sex','age','heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate' , 'temperature', 'spo2', 'urineoutput', 'gcs_value', 'gcs_motor', 'gcs_eyes','ventilation_code']]
target = data['lods']

# # Calculate Pearson Correlation
# correlations = []
# for feature in features.columns:
#     correlation, _ = pearsonr(features[feature], target)
#     correlations.append((feature, abs(correlation)))

# # Sort the features based on their correlation strength
# correlations.sort(key=lambda x: x[1], reverse=True)

# print("Feature\t\t\tCorrelation")
# for feature, correlation in correlations:
#     print(f"{feature}\t\t{correlation:.4f}")

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

# Extract the feature names in the ranked order
ranked_features = [feat for feat, _ in correlations]

# Select the top features for training
top_features = ranked_features[:14]  # Adjust the number of top features as needed

# Subset the data with the top features
X_selected = features[top_features]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, target, test_size=0.2, random_state=42)

# Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a Random Forest classifier with 100 trees
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Define the number of folds for cross-validation
k = 10
cv = KFold(n_splits=k, shuffle=True, random_state=42)

# Perform cross-validation
mae_scores = -cross_val_score(rf, X_selected, target, cv=cv, scoring='neg_mean_absolute_error')
rmse_scores = -cross_val_score(rf, X_selected, target, cv=cv, scoring='neg_root_mean_squared_error')

# Print the cross-validation scores
print("Cross-Validation MAE scores:", mae_scores)
print("Average MAE:", mae_scores.mean())

print("Cross-Validation RMSE scores:", rmse_scores)
print("Average RMSE:", rmse_scores.mean())

# # Train the Random Forest classifier
# rf.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = rf.predict(X_test)

# # Evaluate the model's accuracy
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# mae = mean_absolute_error(y_test, y_pred)
# print('Root Mean Squared Error:', rmse)
# print('Mean Absolute Error:', mae)