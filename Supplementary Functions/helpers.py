import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import os
import torch


def feature_importance(learner, df):
    # define and split training/testing data and training the regressor
    features = df.iloc[:, 0:-1]
    target = df.iloc[:, -1]

    learner.fit(features, target)

    importance = learner.feature_importances_
    features = df.columns[0:-1]

    col = zip(features, importance)

    # sort and save features based on its importance into a dataframe
    importance_data = pd.DataFrame(col, columns=['feature', 'importance'])
    importance_data_asc = importance_data.sort_values('importance')

    # plot the importance data
    fig = plt.figure(figsize=(20, 15))
    plt.barh(y=importance_data_asc['feature'], width=importance_data_asc['importance'], height=0.9)

    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.ylabel("Adsorbate Features", fontsize=15)
    plt.xlabel("Feature Importance", fontsize=15)
    plt.title("Feature Importance of Adsorbates", fontsize=15)
    plt.show();


def pearson_correlation(df, last=False):
    if last == False:
        data = df.iloc[:, :20]
    else:
        data = df.iloc[:, :-1]

    corr = data.corr()
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=220),
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        horizontalalignment='right'
    )
    plt.rcParams['figure.figsize'] = (20, 20)



def transform_2D(series, split):
    result = []
    temp = []
    count = 1
    for i in series:
        if count % split != 0:
            temp.append(i)
            count += 1
        else:
            count = 1
            result.append(temp)
            temp = []
            temp.append(i)
            count += 1
    result.append(temp)
    return result


def scaler(df, filename, target):
    base = pd.read_csv(filename)
    base = base.drop(columns=['Adsorbate 1', 'Adsorbate 2', target])
    scales = {}
    for i in range(len(base.columns)):
        name = base.columns[i]
        minimum = min(base.iloc[:, i])
        maximum = max(base.iloc[:, i])
        scales[name] = [minimum, maximum]

    for i in range(len(df.columns)):
        name_1 = df.columns[i]
        if name_1 == target:
            pass
        else:
            for j, row in enumerate(df[name_1]):
                df.iloc[j, i] = (row - scales[name_1][0]) / (scales[name_1][1] - scales[name_1][0])

    return df





def seed_everything(seed=42):
    """"
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
