# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split

# # Assuming df is your DataFrame with the required structure
# df = pd.read_csv('imputed_li.csv')

# # Sort the data by stay_id and hour_num
# # df.sort_values(['stay_id', 'hour_num'], inplace=True)
# grouped = df.groupby('stay_id').apply(lambda x: x.sort_values('hour_num')).reset_index(drop=True)

# first_two_groups = grouped.groupby('stay_id').head(2)

# # Print the first two groups and their data
# for group, data in first_two_groups.groupby('stay_id'):
#     print(f"Group {group}:")
#     print(data)

# # Extract features (columns 6 to 27)
# features = df.iloc[:, 7:28].values  # Assuming 22 columns and 12 timestamps

# # Normalize the features
# scaler = StandardScaler()
# features_normalized = scaler.fit_transform(features)

# # Reshape the features for 1D CNN input
# features_reshaped = features_normalized.reshape(-1, 12, 21)

# # Define the Sequential model with Dense layers
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(12, 21)),  # Input layer with 64 units and ReLU activation
#     Dense(32, activation='relu'),  # Hidden layer with 32 units and ReLU activation
#     Dense(10, activation='softmax')  # Output layer with 10 units (adjust based on your task) and softmax activation
# ])

# # Compile the model using Adam optimizer and appropriate loss function
# model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# # Assuming you have labels/targets for each stay_id group
# # Adjust the labels based on your specific task and data
# # For simplicity, let's assume you have 10 classes and use one-hot encoding
# labels = np.random.randint(0, 10, size=(len(df), 10))  # Adjust based on your task

# # Train the model
# model.fit(features_reshaped, labels, epochs=10, batch_size=32)  # Adjust epochs and batch_size as needed

# feature_extractor_model = Model(inputs=model.input, outputs=model.layers[-2].output)
# features_output = feature_extractor_model.predict(features_reshaped)

# # Combine features with stay_id
# stay_ids = df['stay_id'].values.reshape(-1, 1)
# features_with_stay_id = np.hstack((stay_ids, features_output))

# # Convert to DataFrame for further analysis or saving to a file
# features_df = pd.DataFrame(features_with_stay_id, columns=['stay_id'] + [f'feature_{i+1}' for i in range(features_output.shape[1])])


# data2 = pd.read_csv('data2.csv')

# # ... Preprocess data2.csv and extract relevant features ...
# selected_columns_from_data2 = data2[['column1', 'column2', ...]]  # Replace with the actual column names

# # Assuming cnn_features is a DataFrame
# # Merge cnn_features with selected columns from data2 based on 'stay_id'
# combined_data = pd.merge(features_df, selected_columns_from_data2, on='stay_id')

# # Drop 'stay_id' from the combined data (optional)
# combined_data.drop('stay_id', axis=1, inplace=True)

# # Prepare the features and target
# X = combined_data.values  # Features are now cnn_features and selected columns from data2.csv
# y = combined_data['target_column']  # Adjust based on your target column

# # Split the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the XGBoost model
# xgb_model = XGBClassifier()
# xgb_model.fit(X_train, y_train)

# # Predict the target on the test set
# y_pred = xgb_model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy of XGBoost model: {accuracy}')

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
# from tensorflow.keras.utils import plot_model

# # Define a simple CNN model
# model = Sequential()

# # Add Convolutional layers
# model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(100, 1)))  # Assuming input shape is (100, 1)
# model.add(MaxPooling1D(pool_size=2))

# # Flatten layer to connect with dense layers
# model.add(Flatten())

# # Dense layers
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))  # Output layer (adjust output neurons as needed)

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Visualize the model
# plot_model(model, to_file='cnn_model.png', show_shapes=True)


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Read the CSV file
data1 = pd.read_csv('imputed_li_modified.csv')
data2 = pd.read_csv('data12h.csv')

merged_data = pd.merge(data1, data2, on='stay_id', how='inner')
# 2. Sort the data
# data = data.sort_values(by=['stay_id', 'hour_num'])
merged_data.sort_values(by=['stay_id', 'hour_num'], inplace=True)

unique_stay_ids = merged_data['stay_id'].unique()

# for stay_id in unique_stay_ids[:1]:
#     group_data = merged_data[merged_data['stay_id'] == stay_id]
#     print(f"\nMerged Data for stay_id {stay_id}:\n")
#     print(group_data)
#     save_path = f'saved_data.csv'
    
#     # Save the group data to CSV
#     group_data.to_csv(save_path, index=False)

#     print(f"Data for stay_id {stay_id} saved to {save_path}")

# 3. Divide into groups
groups = [group[1] for group in merged_data.groupby('stay_id', sort=False)]

# 4. Prepare data for CNN
def preprocess_data(df):
    # Select input features (columns 7 to 14) and target variable (column 14)
    X = df.iloc[:, 6:13].values  # Assuming 0-based indexing
    y = df['lods'].iloc[0]   # Assuming 0-based indexing

    return X, y

data_dict = {}
for df in groups:
    df_stayid = df['stay_id'].iloc[0]  # Assuming 'id' is unique for each DataFrame
    X, y = preprocess_data(df)
    data_dict[df_stayid] = {'X': X, 'y': y}

all_x = [entry['X'][0] for entry in data_dict.values()]
all_y = [entry['y'] for entry in data_dict.values()]

# 5. define the model
# Assuming a simple CNN architecture
# input_shape = (12, 8, 1)   
# model = models.Sequential()

# # Conv1D layer with 32 filters, kernel size 3, activation 'relu'
# model.add(layers.Reshape((12, 8, 1), input_shape=input_shape))

# # Conv2D layer with 32 filters, kernel size (3, 3), activation 'relu'
# model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))

# # MaxPooling2D layer with pool size (2, 2)
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# # Flatten the output before feeding it to dense layers
# model.add(layers.Flatten())

# # Dense layer with 64 units and activation 'relu'
# model.add(layers.Dense(64, activation='relu'))

# # Output layer with 1 unit (assuming regression) and linear activation
# model.add(layers.Dense(1, activation='linear'))

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


# Train the model using the organized data
X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=42)

# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
