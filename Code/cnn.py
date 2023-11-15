import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Assuming df is your DataFrame with the required structure
df = pd.read_csv('imputed_li.csv')

# Sort the data by stay_id and hour_num
df.sort_values(['stay_id', 'hour_num'], inplace=True)

# Extract features (columns 6 to 27)
features = df.iloc[:, 7:28].values  # Assuming 22 columns and 12 timestamps

# Normalize the features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Reshape the features for 1D CNN input
features_reshaped = features_normalized.reshape(-1, 12, 21)

# Define the Sequential model with Dense layers
model = Sequential([
    Dense(64, activation='relu', input_shape=(12, 21)),  # Input layer with 64 units and ReLU activation
    Dense(32, activation='relu'),  # Hidden layer with 32 units and ReLU activation
    Dense(10, activation='softmax')  # Output layer with 10 units (adjust based on your task) and softmax activation
])

# Compile the model using Adam optimizer and appropriate loss function
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Assuming you have labels/targets for each stay_id group
# Adjust the labels based on your specific task and data
# For simplicity, let's assume you have 10 classes and use one-hot encoding
labels = np.random.randint(0, 10, size=(len(df), 10))  # Adjust based on your task

# Train the model
model.fit(features_reshaped, labels, epochs=10, batch_size=32)  # Adjust epochs and batch_size as needed

feature_extractor_model = Model(inputs=model.input, outputs=model.layers[-2].output)
features_output = feature_extractor_model.predict(features_reshaped)

# Combine features with stay_id
stay_ids = df['stay_id'].values.reshape(-1, 1)
features_with_stay_id = np.hstack((stay_ids, features_output))

# Convert to DataFrame for further analysis or saving to a file
features_df = pd.DataFrame(features_with_stay_id, columns=['stay_id'] + [f'feature_{i+1}' for i in range(features_output.shape[1])])


data2 = pd.read_csv('data2.csv')

# ... Preprocess data2.csv and extract relevant features ...
selected_columns_from_data2 = data2[['column1', 'column2', ...]]  # Replace with the actual column names

# Assuming cnn_features is a DataFrame
# Merge cnn_features with selected columns from data2 based on 'stay_id'
combined_data = pd.merge(features_df, selected_columns_from_data2, on='stay_id')

# Drop 'stay_id' from the combined data (optional)
combined_data.drop('stay_id', axis=1, inplace=True)

# Prepare the features and target
X = combined_data.values  # Features are now cnn_features and selected columns from data2.csv
y = combined_data['target_column']  # Adjust based on your target column

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# Predict the target on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of XGBoost model: {accuracy}')