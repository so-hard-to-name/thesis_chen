import pandas as pd

# Load your CSV file into a pandas DataFrame
df = pd.read_csv('data_all12h_vitalsign_demo.csv')

# Sort the DataFrame by 'ID' and 'Series'
df.sort_values(by=['stay_id', 'hour_num'], inplace=True)

# Define a function for linear interpolation
def linear_interpolation(group):
    group = group.interpolate(method='linear', limit_direction='both')
    return group

# Apply linear interpolation within each group
imputed_df = df.groupby('stay_id').apply(linear_interpolation)

# Replace the original DataFrame with the imputed values
df.update(imputed_df)

# Save the DataFrame with imputed values to a new CSV file
df.to_csv('imputed_li.csv', index=False)
