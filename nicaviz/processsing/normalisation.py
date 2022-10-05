from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler, OneHotEncoder


def feature_prep_cate_le(xtrain, xtest, categorical_columns):
    categorical_dims = {}
    for col in categorical_columns:
        l_enc = LabelEncoder()
        xtrain[col] = l_enc.fit_transform(xtrain[col].values)
        xtest[col] = l_enc.transform(xtest[col].values)
        categorical_dims[col] = len(l_enc.classes_)
    return xtrain, xtest, categorical_dims


def feature_prep_cate_ohe(xtrain, xtest, categorical_columns):
    ohe_train_all = pd.DataFrame()
    ohe_test_all = pd.DataFrame()
    for col in categorical_columns:
        ohe = OneHotEncoder()
        train_ohe = ohe.fit_transform(
            xtrain[col].values.reshape(-1, 1)).todense()
        test_ohe = ohe.transform(xtest[col].values.reshape(-1, 1)).todense()
        colnames = ['{}_{}'.format(col, x) for x in ohe.categories_[0]]
        ohe_train_all = pd.concat(
            [ohe_train_all, pd.DataFrame(train_ohe, columns=colnames)], axis=1)
        ohe_test_all = pd.concat(
            [ohe_test_all, pd.DataFrame(test_ohe, columns=colnames)], axis=1)
    return ohe_train_all, ohe_test_all


def feature_prep_numeric(xtrain, xtest,  numerical_columns):
    for col in numerical_columns:
        scaler = RS = RobustScaler()
        xtrain[col] = scaler.fit_transform(xtrain[col].values.reshape(-1, 1))
        xtest[col] = scaler.transform(xtest[col].values.reshape(-1, 1))
    return xtrain, xtest
