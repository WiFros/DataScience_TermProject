import pandas as pd
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier


def bestSearch(param, df, target):
    '''
    description : A function that finds the best combination of scale and model with only numeric columns

    :param param: Dictionary data type, 'scaler' and 'model' are key values.
    :param df: Data to scale
    :param target: Column to predict
    :return: Returns the best combination with the highest score.
    '''

    scaler = np.array(param.get('scaler'))
    model = np.array(param.get('model'))
    bestDi = {}

    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)
    for s in scaler:
        X_train_scale, X_test_scale = scaled(X_train, X_test, s)
        for m in model:
            bestDi[s + ", " + m] = predict(m, X_train_scale, X_test_scale, y_train, y_test)
            print(s + ", " + m, bestDi[s + ", " + m])

    return max(bestDi, key=bestDi.get), max(bestDi.values())


def scaled(X_train, X_test, scaler):
    '''
    Description : A function that scales to the scale received as a parameter.

    :param X_train: train data
    :param X_test: test data
    :param scaler: Scaler to use, scaler has 'standard', 'minmax', and 'robust'.
    :return: scaled train data, test data
    '''
    if scaler == "standard":
        stdScaler = StandardScaler()
        X_train_scale = stdScaler.fit_transform(X_train)
        X_test_scale = stdScaler.transform(X_test)
        return X_train_scale, X_test_scale

    elif scaler == "robust":
        robustScaler = RobustScaler()
        X_train_scale = robustScaler.fit_transform(X_train)
        X_test_scale = robustScaler.transform(X_test)
        return X_train_scale, X_test_scale

    elif scaler == "minmax":
        mmScaler = MinMaxScaler()
        X_train_scale = mmScaler.fit_transform(X_train)
        X_test_scale = mmScaler.transform(X_test)
        return X_train_scale, X_test_scale


def encoding(encoder, cols, df):
    '''
    Description:  A function that replaces categorical columns with numeric columns

    :param encoder: Encode to use, encoder has 'labelEncoder', 'oneHotEncoder'
    :param cols: Categorical columns
    :param df: data to encode
    :return: encoded data
    '''
    if (encoder == "labelEncoder"):
        label_df = df.copy()
        for c in cols:
            lb = LabelEncoder()
            lb.fit(list(df[c].values))
            label_df[c] = lb.transform(list(df[c].values))

        return label_df

    elif (encoder == "oneHotEncoder"):
        onehot_df = df.copy()
        for c in cols:
            onehot_df = pd.get_dummies(data=onehot_df, columns=[c])

        return onehot_df


def predict(model, X_train_scale, X_test_scale, y_train, y_test):
    '''
    Description: A function that learns targets using models received with scale and encoded data, and to predict targets with learned models.

    :param model: Model to use for learning, model has '"adaboost", "decisiontree", "bagging", "XGBoost", "randomforest" and "gradient"
    :param X_train_scale: Scale and encoded data for learning
    :param X_test_scale: Data to use for predictions
    :param y_train: Target data for learning
    :param y_test: Target data to use for predictions
    :return: Returns the score of the model.
    '''

    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    if model == "adaboost":
        print("ada start")
        # AdaBoostRegressor
        ada_reg = AdaBoostClassifier()
        ada_param = {
            'n_estimators': [25, 50, 100, 200],
            'learning_rate': [0.01, 0.1]  # 8번 러닝
        }
        ada = GridSearchCV(ada_reg, param_grid=ada_param, cv=kfold, n_jobs=-1)
        ada.fit(X_train_scale, y_train)
        return ada.score(X_test_scale, y_test)

    elif model == "decisiontree":
        print("decision start")
        # DecisionTreeRegressor
        decision_tree_model = DecisionTreeClassifier()
        param_grid = {
            'max_depth': [None, 2, 3, 4, 5, 6]
        }
        gsDT = GridSearchCV(decision_tree_model, param_grid=param_grid, cv=kfold, n_jobs=-1)
        gsDT.fit(X_train_scale, y_train)
        return gsDT.score(X_test_scale, y_test)

    elif model == "bagging":
        print("bagging start")
        # BaggingRegressor
        bagging = BaggingClassifier()
        b_param_grid = {
            'n_estimators': [10, 50, 100],  # 3
            'n_jobs': [-1]

        }
        gsBagging = GridSearchCV(bagging, param_grid=b_param_grid, cv=kfold, n_jobs=-1)
        gsBagging.fit(X_train_scale, y_train)
        return gsBagging.score(X_test_scale, y_test)

    elif model == "XGBoost":
        print("xg start")
        # XGBRegressor
        XGB = XGBClassifier()
        xgb_param_grid = {
            'learning_rate': [0.1, 0.01],
            'max_depth': [1, 5, 10, 50],
        }
        gsXGB = GridSearchCV(XGB, param_grid=xgb_param_grid, cv=kfold, n_jobs=-1)
        gsXGB.fit(X_train_scale, y_train)
        return gsXGB.score(X_test_scale, y_test)

    elif model == "randomforest":
        print("random forest start")
        # RandomForestRegressor
        forest = RandomForestClassifier()
        fo_grid = {
            "n_estimators": [200],
            "criterion": ["entropy"],
            "max_depth": [None, 2, 3, 4, 5],
            'n_jobs': [-1]

        }
        gsRd = GridSearchCV(forest, param_grid=fo_grid, cv=kfold, n_jobs=-1)
        gsRd.fit(X_train_scale, y_train)
        return gsRd.score(X_test_scale, y_test)

    elif model == "gradient":
        print("gradient start")
        # GradientBoostingRegressor
        gbr = GradientBoostingClassifier()
        param = {
            "n_estimators": [25, 50, 100],
            "learning_rate": [0.1, 0.01],
            "subsample": [0.5, 0.01]
            # 27번
        }
        gsGd = GridSearchCV(gbr, param_grid=param, cv=kfold, n_jobs=-1)
        gsGd.fit(X_train_scale, y_train)
        return gsGd.score(X_test_scale, y_test)
    elif model == "KNN":
        print("KNN start")
        # KNeighborsClassifier
        knn = KNeighborsClassifier()
        param = {
            'n_neighbors': list(range(1, 10)),
        }
        gsGd = GridSearchCV(knn, param_grid=param, cv=kfold, n_jobs=-1)
        gsGd.fit(X_train_scale, y_train)
        return gsGd.score(X_test_scale, y_test)
    elif model == "LinearRegression":
        print("LinearRegression start")
        # LinearRegression
        lr = LinearRegression()
        param = {
                      "fit_intercept": [True, False],
                      }
        gsGd = GridSearchCV(lr, param_grid=param, cv=kfold, n_jobs=-1)
        gsGd.fit(X_train_scale, y_train)
        return gsGd.score(X_test_scale, y_test)
