import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Step 1: Load the pre-trained CNN model
model = load_model('cnn_model_li.keras')

# Step 2: Load the dataset
dataset = pd.read_csv('li_merged_oneline.csv')

# Step 3: Initialize an empty list to store processed data
processed_rows = []

# Step 4: Iterate over the rows of the dataset
for index, row in dataset.iterrows():
    # Extract ID and data to process from the current row
    row_id = row['stay_id']  # Assuming 'id' is the name of the ID column
    data_to_process = row[1:85].values
    print(data_to_process)
    
    # Reshape data_to_process if needed (e.g., for CNN input)
    data_to_process_reshaped = data_to_process.reshape(1, 12, 7, 1)
    print(data_to_process_reshaped)

    data_to_process_reshaped = np.asarray(data_to_process_reshaped).astype('float32')
    
    # Process data using the model
    processed_data = model.predict(data_to_process_reshaped)  # Assuming model expects a 2D input
    print(processed_data)
    processed_data = processed_data.reshape(1, )

    # Append the ID and processed data to the list
    processed_row = np.concatenate([[row_id], processed_data.flatten()])
    processed_rows.append(processed_row)

# Step 5: Create a DataFrame from the processed rows
processed_df = pd.DataFrame(processed_rows)

# Step 6: Export processed data along with IDs to a CSV file
processed_df.to_csv('cnn_processed_data_li.csv', index=False, header=['stay_id'] + [f'feature_{i}' for i in range(processed_df.shape[1] - 1)])
