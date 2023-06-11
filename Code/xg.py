import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Read the CSV file into a DataFrame
data = pd.read_csv('data.csv')

# Select the features and target variable
features = data[['age','heart_rate_min', 'heart_rate_max', 'sbp_min', 'sbp_max', 'dbp_min', 'dbp_max', 'mbp_min', 'mbp_max', 'resp_rate_min', 'resp_rate_max' , 'temperature_min', 'temperature_max', 'spo2_min', 'spo2_max', 'urineoutput', 'gcs_value', 'gcs_motor', 'gcs_eyes']]
target = data['lods']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

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

