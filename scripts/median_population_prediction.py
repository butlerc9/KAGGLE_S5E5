import pandas as pd

# find the modal value of the training set
df_train = pd.read_csv(
    './data/raw/train.csv',
    index_col='id'
)

target_column = 'Listening_Time_minutes'

modal_value = df_train[target_column].median()

print()

# use it as a guess for every single training set
df_test = pd.read_csv(
    './data/raw/test.csv',
    index_col='id'
)

df_submission = (
    df_test
        .assign(Listening_Time_minutes = lambda x: modal_value)
        ['Listening_Time_minutes']
)

# write the csv to the submissions folder
df_submission.to_csv('./submissions/median_population_prediction.csv')