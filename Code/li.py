import numpy as np
import pandas as pd

# Sample two-dimensional data with missing values along columns
data = np.array([
  [1, 100.0, 104.0, 102.0, 113.0, 136.0, 124.5, 74.0, 77.0,75.5,84.0,89.0,86.5,14.0,18.0,16.0,36,36,36,100.0,100.0,100.0],
  [2,np.nan,np.nan,np.nan,131,151,141,61,72,66.5000000000000,80,80,80,16,16,16,"37.2800000000000","37.2800000000000","37.2800000000000",np.nan,np.nan,np.nan],
  [3,83,83,83,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,100,100,100],
  [4,83,92,87.5000000000000,109,123,116,55,65,60,71,84,77.5000000000000,14,16,14.6666666666667,"37.5000000000000","37.5000000000000","37.5000000000000",100,100,100],
  [5,103,103,103,111,111,111,56,56,56,71,71,71,20,20,20,np.nan,np.nan,np.nan,100,100,100],
  [6,111,111,111,133,133,133,63,63,63,83,83,83,19,19,19,np.nan,np.nan,np.nan,99,99,99],
  [7,123,123,123,155,155,155,68,68,68,91,91,91,21,21,21,np.nan,np.nan,np.nan,96,96,96],
  [8,128,128,128,122,122,122,67,67,67,83,83,83,21,21,21,"38.2200000000000","38.2200000000000","38.2200000000000",98,98,98],
  [9,123,123,123,136,136,136,67,67,67,87,87,87,22,22,22,np.nan,np.nan,np.nan,96,96,96],
  [10,124,124,124,108,108,108,61,61,61,77,77,77,17,17,17,np.nan,np.nan,np.nan,94,94,94],
  [11,112,116,114,142,169,155.500000000000,68,69,68.5000000000000,91,98,94.5000000000000,13,16,14.5000000000000,np.nan,np.nan,np.nan,97,98,97.5000000000000],
  [12,116,116,116,118,118,118,64,64,64,83,83,83,13,13,13,"37.5000000000000","37.5000000000000","37.5000000000000",95,95,95]
])
data = data.astype(float)
# Function to perform linear interpolation along a 1D array
def linear_interpolation_1d(y):
    not_nan_indices = ~pd.isna(y)
    x = np.arange(len(y))
    y_interp = np.interp(x, x[not_nan_indices], y[not_nan_indices])
    return y_interp

# Perform linear interpolation for each column
for j in range(data.shape[1]):
    data[:, j] = linear_interpolation_1d(data[:, j])

print("Interpolated data:")
for line in data:
    print(*line)
