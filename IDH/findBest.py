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


def predict(model, X_train_scale, X_test_scale, y_train, y_test):
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    if model == "adaboost":
        print("ada start")
        # AdaBoostClassifier
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
        # DecisionTreeClassifier
        decision_tree_model = DecisionTreeClassifier()
        param_grid = {
            'max_depth': [None, 2, 3, 4, 5, 6]
        }
        gsDT = GridSearchCV(decision_tree_model, param_grid=param_grid, cv=kfold, n_jobs=-1)
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
        # GradientBoostingClassifier
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
