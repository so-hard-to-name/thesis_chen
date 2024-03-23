import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

model = xgb.Booster()
model.load_model('xgb_model_pearson.model')

# Step 2: Load the dataset
data = pd.read_csv('data12h_test_dataset.csv')

data['sex'] = np.where(data['gender'] == "M", 0, 1)
data['heart_rate'] = np.where(abs(data['heart_rate_min'] - 80) >= abs(data['heart_rate_max'] - 80), data['heart_rate_min'], data['heart_rate_max'])
data['sbp'] = np.where(abs(data['sbp_min'] - 120) >= abs(data['sbp_max'] - 120), data['sbp_min'], data['sbp_max'])
data['dbp'] = np.where(abs(data['dbp_min'] - 80) >= abs(data['dbp_max'] - 80), data['dbp_min'], data['dbp_max'])
data['mbp'] = np.where(abs(data['mbp_min'] - 100) >= abs(data['mbp_max'] - 100), data['mbp_min'], data['mbp_max'])
data['resp_rate'] = np.where(abs(data['resp_rate_min'] - 14) >= abs(data['resp_rate_max'] - 14), data['resp_rate_min'], data['resp_rate_max'])
data['temperature'] = np.where(abs(data['temperature_min'] - 36) >= abs(data['temperature_max'] - 36), data['temperature_min'], data['temperature_max'])
data['spo2'] = np.where(abs(data['spo2_min'] - 98) >= abs(data['spo2_max'] - 98), data['spo2_min'], data['spo2_max'])

features = data[['heart_rate', 'sbp', 'resp_rate', 'spo2', 'urineoutput', 'gcs_value', 'gcs_motor', 'ventilation_code']]
target = data['lods']

data_selected = features

predictions = model.predict(xgb.DMatrix(data_selected))

rmse = mean_squared_error(target, predictions, squared=False)
print('Root Mean Squared Error:', rmse)
mae = mean_absolute_error(target, predictions)
print('Mean Absolute Error:', mae)

results_df = pd.DataFrame({'True_Label': target, 'Predicted_Label_pearson': predictions})

# Step 3: Export the DataFrame to a file
results_df.to_csv('test_results_pearson.csv', index=False)