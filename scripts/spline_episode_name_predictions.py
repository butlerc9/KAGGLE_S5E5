import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# ——— load ———
df_train = pd.read_csv('./data/raw/train.csv', index_col='id')
df_test  = pd.read_csv('./data/raw/test.csv')

# ——— prepare X/y ———
X_train = df_train[['Episode_Length_minutes']]
y_train = df_train['Listening_Time_minutes']

X_test  = df_test[['Episode_Length_minutes']]

# ——— build pipeline with a mean‐imputer ———
spline_model = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    SplineTransformer(degree=3, n_knots=5, include_bias=False, extrapolation='constant'),
    LinearRegression()
)

scores = -1 * cross_val_score(
    estimator=spline_model,
    X=X_train,
    y=y_train,
    cv=5,
    scoring='neg_root_mean_squared_error'
) 


# ——— fit & predict ———
spline_model.fit(X_train, y_train)
y_pred = spline_model.predict(X_test)

predictions = pd.Series(y_pred)

submission = (
    pd.concat([df_test.id,predictions],axis=1)
        .rename(columns={0:'Listening_Time_minutes'})
)

print(submission)

# write the csv to the submissions folder
submission.to_csv(
    './submissions/spline_episode_name_prediction.csv',
    index=False
)
