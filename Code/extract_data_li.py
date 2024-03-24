import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Step 1: Load the pre-trained CNN model
model = load_model('cnn_model_li_2.keras')

# Step 2: Load the dataset
dataset = pd.read_csv('li_train_dataset.csv')

# Step 3: Initialize an empty list to store processed data
processed_rows = []

# Step 4: Iterate over the rows of the dataset
for index, row in dataset.iterrows():
    # Extract ID and data to process from the current row
    row_id = row['stay_id']  # Assuming 'id' is the name of the ID column
    data_to_process = row[1:85].values
        
    # Reshape data_to_process if needed (e.g., for CNN input)
    data_to_process_reshaped = data_to_process.reshape(1, 12, 7, 1)

    data_to_process_reshaped = np.asarray(data_to_process_reshaped).astype('float32')
    
    # Process data using the model
    processed_data = model.predict(data_to_process_reshaped)  # Assuming model expects a 2D input
    
    processed_data = processed_data.reshape(1, 16)
    
    # Append the ID and processed data to the list
    processed_row = np.concatenate([[row_id], processed_data.flatten()])
    processed_rows.append(processed_row)

# Step 5: Create a DataFrame from the processed rows
processed_df = pd.DataFrame(processed_rows)

# Step 6: Export processed data along with IDs to a CSV file
processed_df.to_csv('cnn_processed_data_li.csv', index=False, header=['stay_id'] + [f'feature_{i}' for i in range(processed_df.shape[1] - 1)])


csv1 = pd.read_csv('cnn_processed_data_li.csv')
csv2 = pd.read_csv('data12h.csv')

data = pd.merge(csv1, csv2, on='stay_id', how='inner')

features = data[['feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_15', 'urineoutput', 'gcs_value', 'gcs_eyes','gcs_motor', 'ventilation_code']]
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
top_features = ranked_features[:12]  # Adjust the number of top features as needed

# Subset the data with the top features
X_selected = features[top_features]

X_train, X_test, y_train, y_test = train_test_split(X_selected, target, test_size=0.125, random_state=42)

# Create the XGBoost model
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3)

# Train the model
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('Root Mean Squared Error:', rmse)
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

heatmap, xedges, yedges = np.histogram2d(y_test, y_pred, bins=10)

# Normalize the heatmap values to the range [0, 1]
heatmap = heatmap.T / np.max(heatmap)

# Plot the actual vs predicted values with transparency based on point density
plt.scatter(y_test, y_pred, alpha=heatmap.flatten())

# Add a diagonal line representing perfect prediction
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')


plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Actual Plot')
plt.show()