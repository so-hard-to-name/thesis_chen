# import pandas as pd
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

# # Load your CSV file into a pandas DataFrame
# file_path = 'data_all12h_vitalsign_demo.csv'
# df = pd.read_csv(file_path)

# # Create a copy of the DataFrame to store imputed values
# imputed_df = df.copy()

# # Define the columns with missing data (columns 6 to 27)
# columns_with_missing_data = df.columns[6:27]

# # Initialize the IterativeImputer
# imputer = IterativeImputer(random_state=0)

# # Perform multiple imputations for each missing column
# for col in columns_with_missing_data:
#     # Extract the columns needed for imputation
#     cols_for_imputation = df.drop(columns=[col] + list(columns_with_missing_data))
    
#     # Fit the imputer on available data
#     imputer.fit(cols_for_imputation)
    
#     # Impute missing values
#     imputed_values = imputer.transform(cols_for_imputation)
    
#     # Update the imputed values in the imputed_df
#     imputed_df[col] = imputed_values[:, columns_with_missing_data.get_loc(col)]

# # imputed_df now contains the original data with imputed values
# # You can save this DataFrame back to a CSV file if needed
# imputed_df.to_csv('imputed_data.csv', index=False)

#Adjusting the imputation model can be done by modifying the parameters and characteristics of the IterativeImputer from scikit-learn. Here are some ways to adjust the model:

#Number of Iterations: You can adjust the number of iterations the imputer goes through to improve imputation quality. The max_iter parameter controls this. By default, it's set to 10. Increasing it may lead to better results but could also increase computation time.
# imputer = IterativeImputer(max_iter=20, random_state=0)

# Convergence Tolerance: The tol parameter controls the convergence tolerance, which determines when the imputation process stops. Smaller values make the imputer converge more precisely, but they may also increase computation time.
# imputer = IterativeImputer(tol=1e-3, random_state=0)

# Imputation Method: The default imputation method in scikit-learn's IterativeImputer is BayesianRidge. However, you can also try other imputation methods like LinearRegression, DecisionTreeRegressor, etc. by specifying the estimator parameter.
# from sklearn.linear_model import LinearRegression

# imputer = IterativeImputer(estimator=LinearRegression(), random_state=0)

# Random Seed: Setting the random_state parameter to a specific seed ensures reproducibility. If you want the same imputation results each time you run the code, specify a fixed seed.
# imputer = IterativeImputer(random_state=42)

# Normalization: Depending on your data, it might be beneficial to normalize or scale your features before imputation. You can do this using preprocessing techniques.

# Feature Selection: You can also experiment with feature selection techniques to determine which features to include in the imputation process. Removing irrelevant or highly correlated features may improve imputation quality.

# Evaluate and Fine-Tune: After adjusting the model, it's essential to evaluate its performance. You can use metrics like mean absolute error (MAE) or mean squared error (MSE) to assess how well the imputed values match the true values. If necessary, further fine-tune the model based on the evaluation results.

# Remember that the choice of model and parameters may depend on the specific characteristics of your dataset and the nature of missing data. It's a good practice to experiment with different configurations and assess their performance to find the best approach for your particular problem.

# The IterativeImputer is a class provided by scikit-learn, a popular machine learning library in Python, that is used for imputing missing values in a dataset. It is part of scikit-learn's imputation module and is designed to handle missing data by iteratively estimating missing values based on the values of other features in the dataset. This imputation method is often referred to as Multiple Imputation by Chained Equations (MICE).

# Here's how IterativeImputer works:

# It takes a dataset with missing values as input.

# For each feature with missing values, it treats that feature as the target variable to be imputed while using all other features as predictors.

# It iteratively models and imputes missing values for each feature. In each iteration, it uses a regression model (such as BayesianRidge, LinearRegression, DecisionTreeRegressor, etc.) to estimate the missing values for a specific feature based on the values of the other features.

# The process repeats for a specified number of iterations or until convergence (controlled by the max_iter and tol parameters).

# The imputer returns the dataset with the missing values replaced with the imputed values.

# The IterativeImputer is a flexible and powerful tool for handling missing data because it leverages relationships between features to make informed imputations. It's particularly useful when data is missing at random and when the missingness mechanism is not completely arbitrary.

# Here's a simplified example of how to use IterativeImputer to impute missing values:

# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

# # Create an IterativeImputer instance
# imputer = IterativeImputer(max_iter=10, random_state=0)

# # Fit and transform the imputer on your dataset
# imputed_data = imputer.fit_transform(your_dataset_with_missing_values)


import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load your CSV file into a pandas DataFrame
file_path = 'data_all12h_vitalsign_demo.csv'
df = pd.read_csv(file_path)

# Columns to impute (column 6 to column 27)
columns_to_impute = df.columns[6:27]

# Create a mask for missing values in the columns to impute
mask = df[columns_to_impute].isnull()

# Initialize the IterativeImputer
imputer = IterativeImputer(max_iter=10, random_state=0)

# Perform imputation only on the selected columns
imputed_values = imputer.fit_transform(df[columns_to_impute])

# Create a new DataFrame with imputed values for the selected columns
imputed_df = df.copy()
imputed_df[columns_to_impute] = imputed_values

# imputed_df now contains the original data with imputed values for the selected columns
# You can save this DataFrame back to a CSV file if needed
imputed_df.to_csv('imputed_data.csv', index=False)
