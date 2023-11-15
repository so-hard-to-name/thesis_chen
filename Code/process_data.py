import pandas as pd

# Load the CSV files into DataFrames
data1 = pd.read_csv('data_all12h_vitalsign.csv')
data2 = pd.read_csv('data12h.csv')

# Extract the unique stay_ids from each DataFrame
stay_ids_data1 = set(data1['stay_id'])
stay_ids_data2 = set(data2['stay_id'])

# Find the common stay_ids
common_stay_ids = stay_ids_data1.intersection(stay_ids_data2)

# Filter data1 to keep all rows with common stay_ids
filtered_data1 = data1[data1['stay_id'].isin(common_stay_ids)]

# Filter data2 to keep rows with common stay_ids
filtered_data2 = data2[data2['stay_id'].isin(common_stay_ids)]

# Save the filtered DataFrames to new CSV files
filtered_data1.to_csv('filtered_data1.csv', index=False)
filtered_data2.to_csv('filtered_data2.csv', index=False)


# # this will remove those record that the whole column has no data

# import pandas as pd

# # Load the CSV file into a DataFrame
# # Replace 'your_file.csv' with the actual file path
# df = pd.read_csv('imputed_li.csv')

# # Group by 'stay_id'
# grouped = df.groupby('stay_id')

# # Initialize an empty list to store valid groups
# valid_groups = []

# # Iterate through each group
# for stay_id, group in grouped:
#     # Check if any column from 6 to 27 has any non-missing data in the group
#     if group.iloc[:, 6:28].notna().all().all():
#         # If there's non-missing data, add the group to the valid groups list
#         valid_groups.append(group)

# # Concatenate the valid groups back into a DataFrame
# valid_data = pd.concat(valid_groups)

# # Save the processed data to a new CSV file
# # Replace 'processed_data.csv' with the desired output file path
# valid_data.to_csv('processed_data.csv', index=False)
