import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier
from yellowbrick import ROCAUC
from yellowbrick.classifier import ConfusionMatrix


def FindBest(param, df, target):
    # FindBest function find best score of scaler and fitting model

    # Parameters
    # param : list of scaler and model
    # df : DataFrame
    # target : Column what we want to predict

    scaler = np.array(param.get('scaler'))
    model = np.array(param.get('model'))
    bestDi = {}
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)
    for s in scaler:
        X_train_scale, X_test_scale = scaled(X_train, X_test, s)
        for m in model:
            temp, file_path = scoring(m, X_train_scale, X_test_scale, y_train, y_test,s)
            bestDi[file_path] = temp
            print(file_path, bestDi[file_path])

    return bestDi,file_path
    #return max(bestDi, key=bestDi.get), max(bestDi.values())


def scaled(X_train, X_test, scaler):

    # scaled function scales train dataset and test dataset as type of scaler

    # Parameters
    # X_train : train dataset
    # X_test : test dataset
    # scaler : list of scaler
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

def predict(model, X_train_scale, X_test_scale, y_train, y_test,scal):

    # predict function predict score each model

    # Parameters
    # model : list of model
    # X_train_scale : scaled train dataset
    # X_test_scale : scaled test dataset
    # y_train : data for leaning
    # y_test : data for predicting

    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    if model == "adaboost":
        # AdaBoostRegressor
        ada_reg = AdaBoostClassifier()
        ada_param = {
            'n_estimators': [25, 50, 100, 200],
            'learning_rate': [0.01, 0.1] #8번 러닝
        }

        ada = GridSearchCV(ada_reg, param_grid=ada_param, cv=kfold,n_jobs=-1)
        ada.fit(X_train_scale, y_train)
        joblib.dump(ada,'./'+model+'_model_'+scal+'.pkl')
        return ada.score(X_test_scale, y_test)

    elif model == "decisiontree":
        # DecisionTreeRegressor
        decision_tree_model = DecisionTreeClassifier()
        param_grid = {
            'max_depth': [None, 2, 3, 4, 5, 6]
        }
        gsDT = GridSearchCV(decision_tree_model, param_grid=param_grid, cv=kfold,n_jobs=-1)
        gsDT.fit(X_train_scale, y_train)
        joblib.dump(gsDT,'./'+model+'_model_'+scal+'.pkl')
        return gsDT.score(X_test_scale, y_test)

    elif model == "bagging":
        # BaggingRegressor
        bagging = BaggingClassifier()
        b_param_grid = {
            'n_estimators': [10, 50, 100],#3
            'n_jobs' : [-1]

        }
        gsBagging = GridSearchCV(bagging, param_grid=b_param_grid, cv=kfold,n_jobs=-1)
        gsBagging.fit(X_train_scale, y_train)
        joblib.dump(gsBagging,'./'+model+'_model_'+scal+'.pkl')
        return gsBagging.score(X_test_scale, y_test)

    elif model == "XGBoost":
        # XGBRegressor
        XGB = XGBClassifier()
        xgb_param_grid = {
            'learning_rate': [0.1, 0.01],
            'max_depth': [5, 10, 50],
        }
        gsXGB = GridSearchCV(XGB, param_grid=xgb_param_grid, cv=kfold,n_jobs=-1)
        gsXGB.fit(X_train_scale, y_train)
        joblib.dump(gsXGB,'./'+model+'_model_'+scal+'.pkl')
        return gsXGB.score(X_test_scale, y_test)

    elif model == "randomforest":
        # RandomForestRegressor
        forest = RandomForestClassifier()
        fo_grid = {
            "n_estimators": [200],
            "criterion": ["entropy"],
            "max_depth": [None, 2, 3, 4, 5],
            'n_jobs' : [-1]

        }
        gsRd = GridSearchCV(forest, param_grid=fo_grid, cv=kfold,n_jobs=-1)
        gsRd.fit(X_train_scale, y_train)
        joblib.dump(gsRd,'./'+model+'_model_'+scal+'.pkl')
        return gsRd.score(X_test_scale, y_test)

    elif model == "gradient":
        # GradientBoostingRegressor
        gbr = GradientBoostingClassifier()
        param = {
            "n_estimators": [25, 50, 100],
            "learning_rate": [0.1, 0.01],
            "subsample": [0.5, 0.01]
            # 27번
        }
        gsGd = GridSearchCV(gbr, param_grid=param, cv=kfold,n_jobs=-1)
        gsGd.fit(X_train_scale, y_train)
        joblib.dump(gsGd,'./'+model+'_model_'+scal+'.pkl')

        return gsGd.score(X_test_scale, y_test)
    elif model == "KNN":
        # NearestNeighborsClassification
        knn = KNeighborsClassifier()
        param = {
            "n_neighbors": [3, 5, 7],
            "n_jobs" : [-1]
        }
        gsKN = GridSearchCV(knn, param_grid=param, cv=kfold,n_jobs=-1)
        gsKN.fit(X_train_scale, y_train)
        joblib.dump(gsKN,'./'+model+'_model_'+scal+'.pkl')
        return gsKN.score(X_test_scale, y_test)

def scoring(model, X_train_scale, X_test_scale, y_train, y_test,scal):
        model_score = joblib.load('./Model/'+model+'_model_'+scal+'.pkl')
        model_path = model+'_model_'+scal

        cm = ConfusionMatrix(model_score, classes=['GALAXY', 'QSO', 'STAR'])
        cm.fit(X_train_scale, y_train)
        predicted = model_score.predict(X_test_scale)
        cm.title = model +"-"+ scal
        cm.score(X_test_scale, y_test)
        #cm.show()
        cm.finalize()
        plt.savefig(model +'-'+ scal+'.png')
        plt.clf()
        print(classification_report(y_test, predicted))

        roc = ROCAUC(model_score, classes=['GALAXY', 'QSO', 'STAR'])
        roc.fit(X_train_scale, y_train)
        roc.score(X_test_scale, y_test)
        roc.title = model + scal
        #roc.show()
        roc.finalize()
        plt.savefig(model +'-'+ scal+'.png')
        plt.clf()
        return model_score.score(X_test_scale, y_test),model_path