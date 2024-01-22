import pandas as pd

# Load CSV files
csv1 = pd.read_csv('imputed_li_modified.csv')
csv2 = pd.read_csv('data12h.csv')

# Select only the relevant columns from csv1
# csv1 = csv1.iloc[:, -7:]

# Merge dataframes on the 'stay_id' column
merged_data = pd.merge(csv1, csv2, on='stay_id', how='inner')

# Now merged_data contains the relevant data from both CSV files

# Optionally, you can sort the data by 'stay_id' and 'num' within each group
merged_data.sort_values(by=['stay_id', 'hour_num'], inplace=True)

# Assuming merged_data is already defined

# Get unique stay_ids
unique_stay_ids = merged_data['stay_id'].unique()

# Print data for the first two groups
for stay_id in unique_stay_ids[:2]:
    group_data = merged_data[merged_data['stay_id'] == stay_id]
    print(f"\nMerged Data for stay_id {stay_id}:\n")
    print(group_data)
    save_path = f'saved_data.csv'
    
    # Save the group data to CSV
    group_data.to_csv(save_path, index=False)

    print(f"Data for stay_id {stay_id} saved to {save_path}")
