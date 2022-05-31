import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier,\
    BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from xgboost import XGBClassifier,XGBRegressor
from yellowbrick import ROCAUC
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.regressor import prediction_error


def bestSearch(mode, param, df, target):
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

    if mode == "score":
        print("---------------scoring start-------------------")
        for s in scaler:
            X_train_scale, X_test_scale = scaled(X_train, X_test, s)
            for m in model:
                temp, file_path = scoring(m, X_train_scale, X_test_scale, y_train, y_test,s,"Regressor")
                bestDi[file_path] = temp
                print(file_path, bestDi[file_path])
    elif mode == "fit":
        print("---------------fitting start-------------------")
        for s in scaler:
            X_train_scale, X_test_scale = scaled(X_train, X_test, s)
            for m in model:
                temp, file_path = predict(m, X_train_scale, X_test_scale, y_train, y_test,s,"Regressor")
                bestDi[file_path] = temp
                print(file_path, bestDi[file_path])

    return bestDi,file_path
    #return max(bestDi, key=bestDi.get), max(bestDi.values())

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


def predict(model, X_train_scale, X_test_scale, y_train, y_test,scal,mode):
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
    model_path = model + '_model_' + scal

    if model == "adaboost":
        if mode =="Classifier":
            ada_reg = AdaBoostClassifier()
        elif mode == "Regressor":
            ada_reg = AdaBoostRegressor()
        ada_param = {
            'n_estimators': [25, 50, 100, 200],
            'learning_rate': [0.01, 0.1] #8번 러닝
        }

        ada = GridSearchCV(ada_reg, param_grid=ada_param, cv=kfold,n_jobs=-1)
        ada.fit(X_train_scale, y_train)
        joblib.dump(ada,'./'+model+'_model_'+scal+'.pkl')

        print(ada.best_params_)
        return ada.score(X_test_scale, y_test),model_path

    elif model == "decisiontree":
        if mode == "Classifier":
            decision_tree_model = DecisionTreeClassifier()
        elif mode == "Regressor":
            decision_tree_model = DecisionTreeRegressor()
        param_grid = {
            'max_depth': [None, 2, 3, 4, 5, 6]
        }
        gsDT = GridSearchCV(decision_tree_model, param_grid=param_grid, cv=kfold,n_jobs=-1)
        gsDT.fit(X_train_scale, y_train)
        joblib.dump(gsDT,'./'+model+'_model_'+scal+'.pkl')

        print(gsDT.best_params_)
        return gsDT.score(X_test_scale, y_test),model_path

    elif model == "bagging":
        if mode == "Classifier":
            bagging = BaggingClassifier()
        elif mode == "Regressor":
            bagging = BaggingRegressor()

        b_param_grid = {
            'n_estimators': [10, 50, 100],#3
            'n_jobs' : [-1]

        }
        gsBagging = GridSearchCV(bagging, param_grid=b_param_grid, cv=kfold,n_jobs=-1)
        gsBagging.fit(X_train_scale, y_train)
        joblib.dump(gsBagging,'./'+model+'_model_'+scal+'.pkl')

        print(gsBagging.best_params_)
        return gsBagging.score(X_test_scale, y_test),model_path

    elif model == "XGBoost":
        if mode == "Classifier":
            XGB = XGBClassifier()
        elif mode == "Regressor":
            XGB = XGBRegressor()
        xgb_param_grid = {
            'learning_rate': [0.2,0.1, 0.01],
            'max_depth': [5, 10, 20],
        }
        gsXGB = GridSearchCV(XGB, param_grid=xgb_param_grid, cv=kfold,n_jobs=-1)
        gsXGB.fit(X_train_scale, y_train)
        joblib.dump(gsXGB,'./'+model+'_model_'+scal+'.pkl')

        print(gsXGB.best_params_)
        return gsXGB.score(X_test_scale, y_test),model_path

    elif model == "randomforest":
        if mode == "Classifier":
            forest = RandomForestClassifier()
        elif mode == "Regressor":
            forest = RandomForestRegressor()
        fo_grid = {
            "n_estimators": [200],
            "max_depth": [None, 2, 3, 4, 5],
            'n_jobs' : [-1]

        }
        gsRd = GridSearchCV(forest, param_grid=fo_grid, cv=kfold,n_jobs=-1)
        gsRd.fit(X_train_scale, y_train)
        joblib.dump(gsRd,'./'+model+'_model_'+scal+'.pkl')

        print(gsRd.best_params_)
        return gsRd.score(X_test_scale, y_test),model_path

    elif model == "gradient":
        if mode == "Classifier":
            gbr = GradientBoostingClassifier()
        elif mode == "Regressor":
            gbr = GradientBoostingRegressor()
        param = {
            "n_estimators": [25, 50, 100],
            "learning_rate": [0.1, 0.01],
            "subsample": [0.5, 0.01]
            # 27번
        }
        gsGd = GridSearchCV(gbr, param_grid=param, cv=kfold,n_jobs=-1)
        gsGd.fit(X_train_scale, y_train)
        joblib.dump(gsGd,'./'+model+'_model_'+scal+'.pkl')

        print(gsGd.best_params_)
        return gsGd.score(X_test_scale, y_test),model_path
    elif model == "KNN":
        if mode == "Classifier":
            knn = KNeighborsClassifier()
        elif mode == "Regressor":
            knn = KNeighborsRegressor()
        param = {
            "n_neighbors": [3, 5, 7],
            "n_jobs" : [-1]
        }
        gsKN = GridSearchCV(knn, param_grid=param, cv=kfold,n_jobs=-1)
        gsKN.fit(X_train_scale, y_train)
        joblib.dump(gsKN,'./'+model+'_model_'+scal+'.pkl')

        print(gsKN.best_params_)
        return gsKN.score(X_test_scale, y_test),model_path

def scoring(model, X_train_scale, X_test_scale, y_train, y_test,scal,mode):
        model_score = joblib.load('./Model/'+model+'_model_'+scal+'.pkl')
        model_score = joblib.load('./Model_Over_Regressor/'+model+'_model_'+scal+'.pkl')
        model_path = model+'_model_'+scal

        if  mode == "Classifier":
            cm = ConfusionMatrix(model_score, classes=['GALAXY', 'QSO', 'STAR'])
            cm.fit(X_train_scale, y_train)
            predicted = model_score.predict(X_test_scale)
            cm.title = model +"-"+ scal
            cm.score(X_test_scale, y_test)
            #cm.show()
            cm.finalize()
            plt.savefig(model +'-'+ scal+'.png')
            plt.savefig(model +'-'+ scal+'Confusion'+'.png')
            plt.clf()
            print(classification_report(y_test, predicted))
            roc = ROCAUC(model_score, classes=['GALAXY', 'QSO', 'STAR'])
            roc.fit(X_train_scale, y_train)
            roc.score(X_test_scale, y_test)
            roc.title = model + scal
            roc.title = model + "-" + scal
            # roc.show()
            roc.finalize()
            plt.savefig(model + '-' + scal + '.png')
            plt.savefig(model + '-' + scal + 'ROC' + '.png')
            plt.clf()
        elif mode == "Regressor":
            pe = prediction_error(model_score,X_train_scale,y_train,X_test_scale,y_test)

        return model_score.score(X_test_scale, y_test),model_path