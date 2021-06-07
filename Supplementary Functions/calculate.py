import pandas as pd
import numpy as np
from tqdm.auto import  tqdm
import time

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split

def val_score(df, n, regressors, results_df, num_splits):
    # Define the features and targets
    features = df.iloc[:, 0:-1]
    target = df.iloc[:, -1]
    num_features = len(features.columns)

    # Obtain and save the cross_val_score for each regressor
    results = []
    fold = RepeatedKFold(n_splits=num_splits, n_repeats=5, random_state=42)
    start = time.time()
    results.append(num_features)
    for i in tqdm(regressors):
        temp = []

        mae = cross_val_score(i, features, target, cv=fold, scoring='neg_mean_absolute_error')
        temp.append(np.abs(mae).mean().round(4))

        r2 = cross_val_score(i, features, target, cv=fold, scoring='neg_root_mean_squared_error')
        temp.append(np.abs(r2).mean().round(4))
        
        results.append(temp)
    end = time.time()

    training_time = end - start

    # Save all data to our dataframe
    results.append(training_time)
    results_df.loc[n] = results

def ml_predict(dataframe, name, reg, new_col, final_col):
    dataframe[new_col] = np.nan
    for i, col in enumerate(dataframe[new_col]):
        if pd.isnull(col):
            dataframe.iloc[i, -1] = reg.predict(pd.DataFrame(dataframe.iloc[i, 0:-2]).T)

    dataframe[final_col] = dataframe[name]
    for i, col in enumerate(dataframe[final_col]):
        if pd.isnull(col):
            dataframe.iloc[i, -1] = reg.predict(pd.DataFrame(dataframe.iloc[i, 0:-3]).T)

def calc_average(vals, split):
    temp = []
    averages = []
    count = 1
    for i in vals:
        if count % split != 0:
            temp.append(i)
            count += 1
        else:
            count = 1
            avg = sum(temp) / len(temp)
            averages.append(avg)
            temp = []
            temp.append(i)
            count += 1
    avg = sum(temp) / len(temp)
    averages.append(avg)
    return averages
