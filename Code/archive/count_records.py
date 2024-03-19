import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('data12h.csv')

for i in range (25):
    count = len(df[df['lods'] == i])
    print("Number of rows with '{}' in column '{}': {}".format(i, 'lods', count))

