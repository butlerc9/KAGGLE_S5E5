from scripts.data_processing import preprocessing, postprocessing
import pandas as pd

# machine learning libraries
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_log_error
import lightgbm as lgb
import optuna


def run_cross_validation(df_train: pd.DataFrame, hyperparameters:dict, TARGET_COLUMN: str) -> float:

    NUMBER_OF_SPLITS = 4
        
    outer_kfold = KFold(n_splits=NUMBER_OF_SPLITS)

    list_train_scores = []
    list_test_scores = []

    print(f'Starting training...')
    print(f'Train data shape: ', df_train.shape)
    print('Number of folds: ', NUMBER_OF_SPLITS)
    print('\n')

    for fold_number, (infold_training_indices, infold_test_indices) in enumerate(outer_kfold.split(df_train), 1):

        # Pre-processing of training data in kfold
        X_train = df_train.loc[infold_training_indices,df_train.columns != TARGET_COLUMN]
        y_train = df_train.loc[infold_training_indices,TARGET_COLUMN]

        X_train = preprocessing(X_train)
        X_train = postprocessing(X_train)

        # Pre-processing of training data in kfold for in-fold validation
        X_test = df_train.loc[infold_test_indices,df_train.columns != TARGET_COLUMN]
        y_test = df_train.loc[infold_test_indices,TARGET_COLUMN]
        
        X_test = preprocessing(X_test)
        X_test = postprocessing(X_test)

        model = lgb.LGBMRegressor(
            **hyperparameters
        )

        model.fit(
            X=X_train,
            y=y_train,
            eval_set=[(X_test,y_test)],
            # callbacks=[lgb.early_stopping(stopping_rounds=25,verbose=False)]
        )

        y_train_preds = model.predict(X_train)
        train_rmse = root_mean_squared_log_error(y_true=y_train,y_pred=y_train_preds)
        list_train_scores.append(train_rmse)

        y_test_preds = model.predict(X_test)
        test_rmse = root_mean_squared_log_error(y_true=y_test,y_pred=y_test_preds)
        list_test_scores.append(test_rmse)

        print(f'--- Fold {fold_number} Completed ---')
        print(f'train_rmse, test_rmse - {train_rmse:.3f},{test_rmse:.3f}')

    print('\n')
    print('--- Training_Completed ---')
    average_train_cv_score = sum(list_train_scores)/len(list_train_scores)
    print(f'The average train cv_score is {average_train_cv_score:.3f}')
    average_test_cv_score = sum(list_test_scores)/len(list_test_scores)
    print(f'The average test cv_score is {average_test_cv_score:.3f}')

    return average_test_cv_score