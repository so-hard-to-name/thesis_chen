import pandas as pd

# Load the dataset from CSV
data = pd.read_csv('li_merged_oneline.csv')

# Shuffle the dataset to randomize the rows
data = data.sample(frac=1).reset_index(drop=True)

# Calculate the split index
split_index = int(0.8 * len(data))

# Split the dataset
train_data = data[:split_index]
test_data = data[split_index:]

# Save the split datasets to separate CSV files
train_data.to_csv('li_train_dataset.csv', index=False)
test_data.to_csv('li_test_dataset.csv', index=False)
