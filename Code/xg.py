import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# Read the CSV file into a DataFrame
data = pd.read_csv('data24h_abs.csv')

# Select the features and target variable
# features = data[['age','heart_rate_min', 'heart_rate_max', 'sbp_min', 'sbp_max', 'dbp_min', 'dbp_max', 'mbp_min', 'mbp_max', 'resp_rate_min', 'resp_rate_max' , 'temperature_min', 'temperature_max', 'spo2_min', 'spo2_max', 'urineoutput', 'gcs_value', 'gcs_motor', 'gcs_eyes', 'ventilation_code']]
features = data[['age','heart_rate_abs', 'sbp_abs', 'dbp_abs', 'mbp_abs', 'resp_rate_abs' , 'temperature_abs', 'spo2_abs', 'urineoutput', 'gcs_value', 'gcs_motor', 'gcs_eyes', 'ventilation_code']]
target = data['lods']

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
top_features = ranked_features[:10]  # Adjust the number of top features as needed

# Subset the data with the top features
X_selected = features[top_features]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, target, test_size=0.2, random_state=42)

# Create the XGBoost model
xgb_model = xgb.XGBRegressor()

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('Root Mean Squared Error:', rmse)
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

