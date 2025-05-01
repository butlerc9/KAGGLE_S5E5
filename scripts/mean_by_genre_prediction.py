import pandas as pd

df_train = pd.read_csv(
    './data/raw/train.csv',
    index_col='id'
)

target_column = 'Listening_Time_minutes'

df_mean_listening_time_by_Genre = (
    df_train
        .loc[:,['Genre',target_column]]
        .groupby(by='Genre')
        .mean()
        .reset_index()
)

df_test = pd.read_csv(
    './data/raw/test.csv'
)

df_submission = (
    df_test
        .merge(
            right=df_mean_listening_time_by_Genre,
            how='inner',
            left_on='Genre',
            right_on='Genre'
        )
        [['id','Listening_Time_minutes']]
)

print(len(df_submission))

# write the csv to the submissions folder
df_submission.to_csv(
    './submissions/mean_by_genre_prediction.csv',
    index=False
)