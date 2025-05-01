# dataframe and data manipulation library
import pandas as pd
from scripts.cross_validation import run_cross_validation, objective
import optuna

def main():

    # hyperparameters = {
    #     "max_depth": -1,
    #     "num_leaves": 1024,
    #     "colsample_bytree": 0.7,
    #     "learning_rate": 0.03,
    #     "max_bin": 1024,
    #     "verbosity":0
    # }

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1)

if __name__ == '__main__':
    main()
