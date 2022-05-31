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


def FindBest(param, df, target):
    # FindBest function find best score of scaler and fitting model

    # Parameters
    # param : list of scaler and model
    # df : DataFrame
    # target : Column what we want to predict

    scaler = np.array(param.get('scaler'))
    model = np.array(param.get('model'))

    BestResult = {}

    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)
    for s in scaler:
        X_train_scale, X_test_scale = scaled(X_train, X_test, s)
        for m in model:
            BestResult[s + ", " + m] = predict(m, X_train_scale, X_test_scale, y_train, y_test)
            print("scaler : " + s + "\n" + "model : " + m, BestResult[s + ", " + m])

    return max(BestResult, key=BestResult.get), max(BestResult.values())


def scaled(X_train, X_test, scaler):

    # scaled function scales train dataset and test dataset as type of scaler

    # Parameters
    # X_train : train dataset
    # X_test : test dataset
    # scaler : list of scaler

    if scaler == "standard":
        stdScaler = StandardScaler()
        X_train_scaled = stdScaler.fit_transform(X_train)
        X_test_scaled = stdScaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    elif scaler == "robust":
        robustScaler = RobustScaler()
        X_train_scaled = robustScaler.fit_transform(X_train)
        X_test_scaled = robustScaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    elif scaler == "minmax":
        mmScaler = MinMaxScaler()
        X_train_scaled = mmScaler.fit_transform(X_train)
        X_test_scaled = mmScaler.transform(X_test)
        return X_train_scaled, X_test_scaled


def predict(model, X_train_scale, X_test_scale, y_train, y_test):

    # predict function predict score each model

    # Parameters
    # model : list of model
    # X_train_scale : scaled train dataset
    # X_test_scale : scaled test dataset
    # y_train : data for leaning
    # y_test : data for predicting

    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    if model == "adaboost":
        print("ada start")
        # AdaBoostClassifier
        ada_reg = AdaBoostClassifier()
        ada_param_grid = {
            'n_estimators': [25, 50, 100, 200],
            'learning_rate': [0.01, 0.1]  # 8번 러닝
        }
        ada = GridSearchCV(ada_reg, param_grid=ada_param_grid, cv=kfold, n_jobs=-1)
        ada.fit(X_train_scale, y_train)
        return ada.score(X_test_scale, y_test)

    elif model == "decisiontree":
        print("decision start")
        # DecisionTreeClassifier
        decision_tree_model = DecisionTreeClassifier()
        dt_param_grid = {
            'max_depth': [None, 2, 3, 4, 5, 6]
        }
        gsDT = GridSearchCV(decision_tree_model, param_grid=dt_param_grid, cv=kfold, n_jobs=-1)
        gsDT.fit(X_train_scale, y_train)
        return gsDT.score(X_test_scale, y_test)

    elif model == "bagging":
        print("bagging start")
        # BaggingClassifier
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
        # XGBClassifier
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
        # RandomForestClassifier
        forest = RandomForestClassifier()
        rf_param_grid = {
            "n_estimators": [200],
            "max_depth": [None, 2, 3, 4, 5],
            'n_jobs': [-1]

        }
        gsRd = GridSearchCV(forest, param_grid=rf_param_grid, cv=kfold, n_jobs=-1)
        gsRd.fit(X_train_scale, y_train)
        return gsRd.score(X_test_scale, y_test)

    elif model == "gradient":
        print("gradient start")
        # GradientBoostingClassifier
        gbr = GradientBoostingClassifier()
        gbr_param_grid = {
            "n_estimators": [25, 50, 100],
            "learning_rate": [0.1, 0.01],
            "subsample": [0.5, 0.01]
        }
        gsGd = GridSearchCV(gbr, param_grid=gbr_param_grid, cv=kfold, n_jobs=-1)
        gsGd.fit(X_train_scale, y_train)
        return gsGd.score(X_test_scale, y_test)
    elif model == "KNN":
        print("KNN start")
        # KNeighborsClassifier
        knn = KNeighborsClassifier()
        knn_param_grid = {
            'n_neighbors': list(range(1, 10)),
        }
        gsGd = GridSearchCV(knn, param_grid=knn_param_grid, cv=kfold, n_jobs=-1)
        gsGd.fit(X_train_scale, y_train)
        return gsGd.score(X_test_scale, y_test)
