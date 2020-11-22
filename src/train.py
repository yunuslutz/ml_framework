import argparse
import os

import config
import model_dispatcher

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold, model):
    # read training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # training data is where kfold is != to provided fold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where kfold == fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop target and create as np array
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values

    # create validation arrays
    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    # initial decision tree classifier
    clf = model_dispatcher.models[model]

    # fit on training data
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    preds = clf.predict(x_valid)

    # calculate and print metrics
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )

if __name__ == "__main__":
    # initialize argparser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )

    # read arguments from command line
    args = parser.parse_args()

    run(
        fold=args.fold,
        model=args.model
    )