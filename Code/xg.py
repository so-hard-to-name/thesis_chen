import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
data = pd.read_csv('data12h.csv')

data['heart_rate'] = np.where(abs(data['heart_rate_min'] - 65) >= abs(data['heart_rate_max'] - 65), data['heart_rate_min'], data['heart_rate_max'])
data['sbp'] = np.where(abs(data['sbp_min'] - 120) >= abs(data['sbp_max'] - 120), data['sbp_min'], data['sbp_max'])
data['dbp'] = np.where(abs(data['dbp_min'] - 80) >= abs(data['dbp_max'] - 80), data['dbp_min'], data['dbp_max'])
data['mbp'] = np.where(abs(data['mbp_min'] - 100) >= abs(data['mbp_max'] - 100), data['mbp_min'], data['mbp_max'])
data['resp_rate'] = np.where(abs(data['resp_rate_min'] - 14) >= abs(data['resp_rate_max'] - 14), data['resp_rate_min'], data['resp_rate_max'])
data['temperature'] = np.where(abs(data['temperature_min'] - 36) >= abs(data['temperature_max'] - 36), data['temperature_min'], data['temperature_max'])
data['spo2'] = np.where(abs(data['spo2_min'] - 98) >= abs(data['spo2_max'] - 98), data['spo2_min'], data['spo2_max'])

# Select the features and target variable
# features = data[['age','heart_rate_min', 'heart_rate_max', 'sbp_min', 'sbp_max', 'dbp_min', 'dbp_max', 'mbp_min', 'mbp_max', 'resp_rate_min', 'resp_rate_max' , 'temperature_min', 'temperature_max', 'spo2_min', 'spo2_max', 'urineoutput', 'gcs_value', 'gcs_motor', 'gcs_eyes', 'ventilation_code']]
features = data[['age','heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate' , 'temperature', 'spo2', 'urineoutput', 'gcs_value', 'gcs_motor', 'ventilation_code']]
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
    print(f"{feature}\t\t{correlation:.4f}")

# Extract the feature names in the ranked order
ranked_features = [feat for feat, _ in correlations]

# Select the top features for training
top_features = ranked_features[:8]  # Adjust the number of top features as needed

# Subset the data with the top features
X_selected = features[top_features]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, target, test_size=0.2, random_state=42)

# Create the XGBoost model
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('Root Mean Squared Error:', rmse)
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

# show tree start from 0
plt.figure(figsize=(15, 10))
xgb.plot_tree(xgb_model, num_trees=2)
plt.show()

# show all trees
# # Get the list of all trees in the model
# trees = xgb_model.get_booster().get_dump()

# # Create subplots grid
# num_trees = len(trees)
# num_rows = int(num_trees / 3) + 1
# fig, axs = plt.subplots(num_rows, 3, figsize=(12, num_rows * 4))

# # Visualize each tree in a subplot
# for i, tree in enumerate(trees):
#     ax = axs[i // 3, i % 3]  # Select the subplot
#     ax.set_title(f'Tree {i+1}')
#     xgb.plot_tree(xgb_model, num_trees=i, ax=ax)

# # Hide empty subplots if any
# if num_trees < num_rows * 3:
#     for i in range(num_trees, num_rows * 3):
#         fig.delaxes(axs[i // 3, i % 3])

# # Adjust the layout and display the plot
# plt.tight_layout()
# plt.show()