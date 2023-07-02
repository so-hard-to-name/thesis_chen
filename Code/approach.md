# Background

Logistic Organ Dysfunction Score is used to measure patients’ severity with the worst value of 11 parameters in the first 24h in ICU stay. It is proved to useful in the first 7 days in ICU. However, seven of the parameters need laboratory experiments.


# Goal of the Research

The goal of this thesis is to propose a machine learning model to predict patients’ LODS with first 12h vital sign data and bedside data in ICU, to increase the usability of LODS. So that care giver can find out patients’ severity in advance and change the treatment if needed.


# Collected data

Age
Heart rate
Systolic blood pressure (sbp)
Diastolic blood pressure (dbp)
Mean blood pressure (mbp)
Respiratory rate
Temperature
SPO2
Glucose
Total urine output
GCS
GCS Motor
GCS Eyes
Ventilation settings (trach > mech vent > NIV > high flow > o2 > None)
Tracheostomy
InvasiveVent
NonInvasiveVent
HFNC
SupplementalOxygen
None
First day LODS


# Evaluation

Root mean square error (RMSE)
Mean absolute error (MAE)



# Research Approach

1. Get data records from MIMIC-IV dataset with settled conditions (in total around 8200 records)
    - Patients’ are adults. (Age > 18)
    - Patients stayed in ICU for more than 24 hours.
    - For each vital sign, there are at least 3 records so that record of min, max, mean can be calculated.
    - No null values in each data (Glucose can be null).
2. Check records of each data to find abnormal data
    - There are only 13 records with Ventilation settings as None. So, train a model without ventilation settings first and add it to check how it affects the result later.
3. Select models, XGBoost and Random Forest.
    - In random forest, all trees are equal with same weight, which will decrease the probability of overfitting
    - In XGBoost, each tree will correct the error from last and weight will be add which will decrease the error.
    - Train those two models separately and use the better one as final model.
4. Set train data and test data to 80% and 20%.
5. Set all data as features to train and LODS as target, and train the model
6. Check the RMSE (1.879) and MAE (1.4409). The reason might be too much features cause noise.
7. Remove the mean value to train, as in OASIS and LODS, only the worst value are used. (Not sure if min or max is the worst, let the model check that)
8. RMSE and MAE are around 2.
9. Include Ventilation settings to check if it affects result. It do affects the result a lot.
    - Set ventilation with code 1-6 by trach > mech vent > NIV > high flow > o2 > None. So teach is 1 and None is 6
10. Start use Pearson Correlation to check the correlation between features and target, and between features -> didn’t find any useful value.
11. Use baseline value to check vital sign data. For each vital sign, the normal value is a range, so take the middle value as baseline value. For example, an adult’s normal heart rate is 60—100, then 80 is the baseline for heart rate
    - Heart rate: 80
    - Sbp: 120
    - Dbp: 80
    - Mbp: 100
    - Respiratory  rate: 14
    - Temperature: 36
    - Spo2: 98
12. Set absolute difference of each vital sign as features, and include ventilation settings by code 1-6 as a feature to train the model.
13. Use Pearson Correlation to find the top correlated features. -> Top 10 was the best feature combination.
14. Modify Tree depth and tree amount to decrease the fitting
15. But the RMSE is larger than 1.9 and MAE is larger than 1.5 (top 10 features, max depth is 3, tree amount is 100)
16. Use the baseline to find the worst value, the bigger absolute difference.
    - abs(data['heart_rate_min'] - 80) and abs(data['heart_rate_max'] - 80)
17. Use the worst value as features.
18. Check the correlation between features, and between features and target.
    - Correlation table (As in correlation.xlsx sheet 1)
    - Correlation of features and target. (As in correlation.xlsx sheet 1)
19. Modify max_depth, tree amount, and top feature amount to get the best performance. Pearson value larger than 0.15 are selected.
20. RMSE is 1.8298 and MAE is 1.422
21. Check Pearson Correlation, I found that correlation between GCS_motor and GCS_eyes is more than 0.7, which means they are strongly correlated. So I decided to use one of them. At the end, gcs_motor is selected, even though it has higher correlation with gcs_value, 0.02 more than gcs_eyes.
22. Change model parameters, train the model again and check the correlation
    - max_depth = 3
    - Tree amount = 100
    - Correlation are in correlation.xlsx sheet 2
    - RMSE = 1.822 and MAE = 1.417
23. Use spearman,
24. Plot actual vs predict, residual (include density)
25. Linear regression
26. Cross validation.


