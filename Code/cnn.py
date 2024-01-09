import pandas as pd
import numpy as np

# 1. Read the CSV file
data = pd.read_csv('imputed_li.csv')

# 2. Sort the data
data = data.sort_values(by=['stay_id', 'hour_num'])

# 3. Divide into groups
groups = [group[1] for group in data.groupby('stay_id', sort=False)]

# 4. Prepare data for CNN
cnn_data = np.array([group.iloc[:, 6:].values for group in groups])

# Now cnn_data contains the data for your CNN model, structured as (1000, 12, 24) assuming 24 columns (30-7+1)
print(cnn_data[0])
print(groups[0]['stay_id'].iloc[0])