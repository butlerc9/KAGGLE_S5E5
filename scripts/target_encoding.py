from sklearn.preprocessing import TargetEncoder

def target_encoding(X_train, y_train, X_test):

    ### TARGET ENCODING
    # Categorical Columns
    categorical_columns = ["Genre","Publication_Day","Episode_Sentiment","Publication_Time","Podcast_Name"]
    categorical_encoded_columns = [column_name + '_TE' for column_name in categorical_columns]

    encoder = TargetEncoder(categories='auto', smooth='auto', cv=5, random_state=42)
    encoder.fit(X_train[categorical_columns], y_train)
    X_train[categorical_encoded_columns] = encoder.transform(X_train[categorical_columns])
    X_test[categorical_encoded_columns] = encoder.transform(X_test[categorical_columns])    

    # # Interaction Columns
    # interaction_features = [
    #     ('Publication_Day','Publication_Time')
    # ]

    # interaction_features_to_be_encoded = []
    # for feature_1, feature_2 in interaction_features:
    #     feature_name = feature_1 + '_' + feature_2 + '_TE'
    #     X_train[feature_name] = (X_train[feature_1].astype('str') + '_' + X_train[feature_2].astype('str')).astype('category')
    #     X_test[feature_name] = (X_test[feature_1].astype('str') + '_' + X_test[feature_2].astype('str')).astype('category')
    #     interaction_features_to_be_encoded.append(feature_name)
    
    # encoder = TargetEncoder(categories='auto', smooth='auto', cv=5, random_state=42)
    # encoder.fit(X_train[interaction_features_to_be_encoded], y_train)
    # X_train[interaction_features_to_be_encoded] = encoder.transform(X_train[interaction_features_to_be_encoded])
    # X_test[interaction_features_to_be_encoded] = encoder.transform(X_test[interaction_features_to_be_encoded])    

    # # Fitting encoder and transforming data

    return X_train, X_test