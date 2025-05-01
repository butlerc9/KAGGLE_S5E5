import pandas as pd

df_train = pd.read_csv(
    './data/raw/train.csv',
    index_col='id'
)

target_column = 'Listening_Time_minutes'

df_mean_listening_time_by_Podcast_Name = (
    df_train
        .loc[:,['Podcast_Name',target_column]]
        .groupby(by='Podcast_Name')
        .mean()
        .reset_index()
)

df_test = pd.read_csv(
    './data/raw/test.csv'
)

df_submission = (
    df_test
        .merge(
            right=df_mean_listening_time_by_Podcast_Name,
            how='inner',
            left_on='Podcast_Name',
            right_on='Podcast_Name'
        )
        [['id','Listening_Time_minutes']]
)

print(len(df_submission))

# write the csv to the submissions folder
df_submission.to_csv(
    './submissions/mean_by_Podcast_Name_prediction.csv',
    index=False
)