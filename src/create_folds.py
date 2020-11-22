# create_folds.py

import numpy as np
import pandas as pd

import config

from sklearn import datasets
from sklearn import model_selection

def create_folds(data: pd.DataFrame):
    # create new column kfold and fill with -1
    data["kfold"] = -1

    #shuffle data and reset index
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    num_bins = int(np.round(1 + np.log2(len(data))))

    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data["target"], bins=num_bins, labels=False
    )

    # initiate kfold class
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, "kfold"] = f
    
    # drop bins column
    data =  data.drop("bins", axis=1)
    # return dataframe with folds
    return data

if __name__ == "__main__":
    # read training data
    df = pd.read_csv(config.PRE_SPLIT_FILE)

    # run create folds and save to new df
    df_folds = create_folds(df)

    #save new df to csv
    df
