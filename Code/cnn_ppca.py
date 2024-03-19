# from tensorflow.keras.utils import plot_model

# # Visualize the model
# plot_model(model, to_file='cnn_model.png', show_shapes=True)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Read the CSV file
# data1 = pd.read_csv('imputed_li_modified.csv')
# data2 = pd.read_csv('data12h.csv')
data = pd.read_csv('li_train_dataset.csv')
# merged_data = pd.merge(data1, data2, on='stay_id', how='inner')
# # 2. Sort the data
# # data = data.sort_values(by=['stay_id', 'hour_num'])
# merged_data.sort_values(by=['stay_id', 'hour_num'], inplace=True)

# unique_stay_ids = merged_data['stay_id'].unique()

# for stay_id in unique_stay_ids[:1]:
#     group_data = merged_data[merged_data['stay_id'] == stay_id]
#     print(f"\nMerged Data for stay_id {stay_id}:\n")
#     print(group_data)
#     save_path = f'saved_data.csv'
    
#     # Save the group data to CSV
#     group_data.to_csv(save_path, index=False)

#     print(f"Data for stay_id {stay_id} saved to {save_path}")

# 3. Divide into groups
# groups = [group[1] for group in merged_data.groupby('stay_id', sort=False)]

# # 4. Prepare data for CNN
# def preprocess_data(df):
#     # Select input features (columns 7 to 14) and target variable (column 14)
#     X = df.iloc[:, 6:13].values  # Assuming 0-based indexing
#     y = df['lods'].iloc[0]   # Assuming 0-based indexing

#     return X, y

# data_dict = {}
# for df in groups:
#     df_stayid = df['stay_id'].iloc[0]  # Assuming 'id' is unique for each DataFrame
#     X, y = preprocess_data(df)
#     data_dict[df_stayid] = {'X': X, 'y': y}

# all_x = [entry['X'][0] for entry in data_dict.values()]
# all_y = [entry['y'] for entry in data_dict.values()]
x, y = data.iloc[:, 1:85].values, data['lods'].values

x_reshaped = x.reshape(x.shape[0], 12, 7, 1)
print(type(x))

X_train, X_test, y_train, y_test = train_test_split(x_reshaped, y, test_size=0.125, random_state=42)

# 5. define the model
# Assuming a simple CNN architecture
model = models.Sequential()

# Conv1D layer with 32 filters, kernel size 3, activation 'relu'
model.add(layers.Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu', input_shape=(12, 7, 1)))

model.add(layers.Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu'))
# Flatten the output before feeding it to dense layers
model.add(layers.Flatten())

# Output layer with 1 unit (assuming regression) and linear activation
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


model.fit(X_train, y_train, epochs=10, batch_size=48, validation_data=(X_test, y_test))

plot_model(model, to_file='cnn_model.png', show_shapes=True)

model_without_layer = models.Sequential()
for layer in model.layers[:-1]:  # Remove the last layer
    model_without_layer.add(layer)

# Save the modified model
model_without_layer.save("cnn_model_li.keras")