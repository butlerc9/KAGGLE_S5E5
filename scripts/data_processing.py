import pandas as pd

def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    
    # Encode sex as binary flag
    gender_mapping = {
        'male': 0,
        'female': 1
    }
    X['is_female'] = X['Sex'].map(gender_mapping)
    X = X.drop(columns=['Sex'])

    return X

def preprocessing(X: pd.DataFrame) -> pd.DataFrame:

    X = feature_engineering(X)

    return X # Enabled this to stop warnings

def postprocessing(X:pd.DataFrame) -> pd.DataFrame:

    columns_to_drop: list[str] = [
        # 'Sex', # Already dropped in feature engineering
        # 'Age',
        # 'Height',
        # 'Weight',
        # 'Duration',
        # 'Heart_Rate',
        # 'Body_Temp',
        # 'Calories' # target variable
    ]

    X = X.drop(columns=columns_to_drop)

    return X