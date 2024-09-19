# DataScience_TermProject
[DataScience course Term Project]

### INTRODUCTION
There are tens of millions of celestial bodies in the universe. Each object will be divided into not only one type, but also galaxies, nebulae, and planets. However, it is not easy to determine the type of celestial body when we use optical telescopes to make observations. 
To find out more about the type, you need to use a giant optical telescope in space, such as the Hubble Space Telescope, or a radio telescope. These telescopes take a long time to observe and analyze celestial bodies. 
Therefore, we want to produce a program that can analyze classified objects that we already know with data that can be observed with optical telescopes.

### 1. Dataset
We use the Stellar Classification Dataset - SDSS17 in the Kaggle. This dataset has 17 features and 1 target data. Ther are all 100,000 data
![image](https://user-images.githubusercontent.com/84771856/171908083-01510598-71be-4b3c-9f50-bbb06058974d.png)
The number of each class

![image](https://user-images.githubusercontent.com/84771856/171908140-ad46592b-0113-428b-b9c6-f82bf9e00b27.png)

### 2. Data preprocessing
- data over sampling
```
def UnderSampling(data, class_name, num_sample):	
    """
    description : A function that divide the sample by class_name and fill each class with the desired number.

    :param data: original dataframe
    :param class_name: Reference data name
    :param num_sample: Number of classes you want to increase
    :return: Returns data frames each increased by num_sample
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError

    div_data = data[data['class'] == class_name]
    left_data = data[data['class'] != class_name]

    div_data = div_data.sample(num_sample)
    sampled_data = pd.concat([div_data, left_data])

    return sampled_data.reset_index(drop=True)

#MAIN
df = []
df.append(pd.read_csv("star_classification.csv"))

#OverSampling
under_data = UnderSampling(df, 'GALAXY', 40000)

over_sampler = SMOTE()
X = under_data.drop(columns='class')
y = under_data['class']

X_over, y_over = over_sampler.fit_resample(X, y)

over_data = pd.concat([X_over, y_over], axis=1)
over_data.to_csv("sampled_data.csv", index=False)
```

Since the data of the original dataset is three times as many as other classes in the Galaxy class, oversampling was used to match the number of other data. 
First we under sampling the data set where class is galaxy. And using SMOTE match the number of classes. SMOTE's operation method is an oversampling method in which samples of classes with a small number of data are imported and a new sample is created by adding arbitrary values to the data.

- over sampling dataset
- 
![image](https://user-images.githubusercontent.com/84771856/171908636-b6608273-43da-4bab-99d2-f10b9d98a698.png)
The number of oversampled data class

![image](https://user-images.githubusercontent.com/84771856/171908658-efca6c75-31ac-4d93-985d-d9e76db8ed0b.png)

- data scaling
```
- def scaled(X_train, X_test, scaler):
    """
    Description : A function that scales to the scale received as a parameter.

    :param X_train: train data
    :param X_test: test data
    :param scaler: Scaler to use, scaler has 'standard', 'minmax', and 'robust'.
    :return: scaled train data, test data
    """
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
     """
    Description:  A function that replaces categorical columns with numeric columns

    :param encoder: Encode to use, encoder has 'labelEncoder', 'oneHotEncoder'
    :param cols: Categorical columns
    :param df: data to encode
    :return: encoded data
     """
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
```
To determine which of each scaler derives the optimal value, we used standard, robust, and minmax scaler. Each scaler was functionalized to prevent the code from increasing complexly.

- data drop
```
def data_exploration(df,name,mode):
    if not isinstance(df, pd.DataFrame):
        raise TypeError
    #data description
    print(df.head())
    print(df.info())
    print(df.describe().T)
    #data count plot about class
    df["class"].value_counts()
    sns.countplot(df["class"], palette="Set3")
    plt.title(name, fontsize=10)
    if mode=='show':
        plt.show()
    else:
        plt.savefig(name+' count plot'+'.png')
        plt.clf()
    # covarience matrix
    f, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap="PuBu", annot=True, linewidths=0.5, fmt='.2f', ax=ax)
    if mode == 'show':
        plt.show()
    else:
        plt.savefig(name +' covariance matrix'+'.png')
        plt.clf()

    # each column's plot graph
    for column in ['alpha', 'delta', 'r', 'i', 'field_ID', 'redshift', 'plate', 'MJD', 'fiber_ID']:
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x=column, kde=True, hue="class")
        plt.title(column)
        if mode == 'show':
            plt.show()
        else:
            plt.savefig(name +' '+column + '.png')
            plt.clf()

def drop_data(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError

    le = LabelEncoder()
    df["class"] = le.fit_transform(df["class"])
    df["class"] = df["class"].astype(int)
    df = df.drop(['obj_ID', 'alpha', 'delta', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'fiber_ID'], axis=1)
    x = df.drop('class', axis=1)
    y = df['class']
    return x,y

def plot(df,column):
    if not isinstance(df, pd.DataFrame):
        raise TypeError

    le = LabelEncoder()
    for i in range(3):
        sns.kdeplot(data=df[df["class"] == i][column], label = le.inverse_transform([i]))
    sns.kdeplot(data=df[column],label = ["All"])
    plt.legend();

def log_plot(df,column):
    if not isinstance(df, pd.DataFrame):
        raise TypeError
    le = LabelEncoder()
    for i in range(3):
        sns.kdeplot(data=np.log(df[df["class"] == i][column]), label = le.inverse_transform([i]))
    sns.kdeplot(data=np.log(df[column]),label = ["All"])
    plt.legend();
```

The association between each feature and target was confirmed using a heat map using a covariance matrix. And using a plot graph, features that are difficult to distinguish classes were dropped.

- heatmap(original)
![image](https://user-images.githubusercontent.com/84771856/171908782-686361ef-d138-4f0c-9520-92e053b95d99.png)

- heatmap(oversampled data)
![image](https://user-images.githubusercontent.com/84771856/171908822-a4620d6a-abc5-4bc1-b1db-64140717b60c.png)

- plot graph(original)
![image](https://user-images.githubusercontent.com/84771856/171908877-045e221a-7f26-40f4-b61d-15612a54941b.png)
![image](https://user-images.githubusercontent.com/84771856/171908888-662baf64-5a33-407e-a0bd-abb1b813d325.png)
![image](https://user-images.githubusercontent.com/84771856/171908897-2c450093-8fd5-4e6e-9a85-54e2dc6e657c.png)
![image](https://user-images.githubusercontent.com/84771856/171908920-7dec86c2-df29-4274-ac4d-e987c2f2bfa2.png)
![image](https://user-images.githubusercontent.com/84771856/171908927-945c587e-a1e2-4a2c-854d-0e84f43b352a.png)
![image](https://user-images.githubusercontent.com/84771856/171908935-9c9cd4b0-b4ac-4073-afbf-9b2afcb68647.png)
![image](https://user-images.githubusercontent.com/84771856/171908941-fb6f0653-1b5c-4d95-b083-9a3e3749ef2c.png)
![image](https://user-images.githubusercontent.com/84771856/171908949-6fae745f-ad77-4d32-af74-0dd571982d9d.png)
![image](https://user-images.githubusercontent.com/84771856/171908959-4bce1566-b947-4d6e-aaad-5734e9732837.png)
![image](https://user-images.githubusercontent.com/84771856/171908964-ab18cbe9-1a64-489c-bbec-b537475ecfca.png)

We determined to drop the feature('obj_ID', 'alpha', 'delta', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'fiber_ID') by referring to the chart above.

- remove outlier data
```
def rem_outliers(df,persentage):
    """
    description : A function that remove outliers from data frames as many as desired percentages

    :param df: original dataframe
    :param persentage: Lower, upper percentage of data to be removed
    :return: Returns the data frames with outliers removed
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError

    for i in df.select_dtypes(include='number').columns:
        qt1 = df[i].quantile(persentage)
        qt3 = df[i].quantile(persentage)
        iqr = qt3 - qt1
        lower = qt1 - (1.5 * iqr)
        upper = qt3 + (1.5 * iqr)
        min_in = df[df[i] < lower].index
        max_in = df[df[i] > upper].index
        df.drop(min_in, inplace=True)
        df.drop(max_in, inplace=True)
    return df
```
The top 25% and the bottom 25% were set as outliers to remove the outliers.

### 3. Modeling
- findbest.py
```
def predict(model, X_train_scale, X_test_scale, y_train, y_test,scal,mode):
     """
    Description: A function that learns targets using models received with scale and encoded data, and to predict targets with learned models.

    :param model: Model to use for learning, model has '"adaboost", "decisiontree", "bagging", "XGBoost", "randomforest" and "gradient"
    :param X_train_scale: Scale and encoded data for learning
    :param X_test_scale: Data to use for predictions
    :param y_train: Target data for learning
    :param y_test: Target data to use for predictions
    :param scal: Scaler type
    :param mode: Regressor or Classifier
    :return: Returns the score of the model.
     """
    if not isinstance(model, str):
        raise TypeError
    if not isinstance(X_test_scale, pd.DataFrame):
        raise TypeError
    if not isinstance(X_test_scale, pd.DataFrame):
        raise TypeError
    if not isinstance(y_train, pd.DataFrame):
        raise TypeError
    if not isinstance(y_test, pd.DataFrame):
        raise TypeError
    if not isinstance(scal, str):
        raise TypeError
    if not isinstance(mode, str):
        raise TypeError

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
     """
    description : Read the fileized model to print out and verify accuracy, and conduct evaluation

   :param model: Model to use for learning, model has '"adaboost", "decisiontree", "bagging", "XGBoost", "randomforest" and "gradient"
    :param X_train_scale: Scale and encoded data for learning
    :param X_test_scale: Data to use for predictions
    :param y_train: Target data for learning
    :param y_test: Target data to use for predictions
    :param scal: Scaler type
    :param mode: Regressor or Classifier
    :return: Returns the score of the model, path file.
     """
    if not isinstance(model, str):
        raise TypeError
    if not isinstance(X_test_scale, pd.DataFrame):
        raise TypeError
    if not isinstance(X_test_scale, pd.DataFrame):
        raise TypeError
    if not isinstance(y_train, pd.DataFrame):
        raise TypeError
    if not isinstance(y_test, pd.DataFrame):
        raise TypeError
    if not isinstance(scal, str):
        raise TypeError
    if not isinstance(mode, str):
        raise TypeError

    model_score = joblib.load('./Model_Over_Regressor/'+model+'_model_'+scal+'.pkl')
    model_path = model+'_model_'+scal

    if  mode == "Classifier":
        cm = ConfusionMatrix(model_score, classes=['GALAXY', 'QSO', 'STAR'])
        cm.fit(X_train_scale, y_train)
        predicted = model_score.predict(X_test_scale)
        cm.title = model +"-"+ scal
        cm.score(X_test_scale, y_test)
        # cm.show()
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
def bestSearch(mode, param, df, target):
    """
    description : A function that finds the best combination of scale and model with only numeric columns

    :param mode: score = Running the scaling() fit = Running the predict()
    :param param: Dictionary data type, 'scaler' and 'model' are key values.
    :param df: Data to scale
    :param target: Column to predict
    :return: Returns the best combination with the highest score.
     """
    if not isinstance(df, pd.DataFrame):
        raise TypeError
    if not isinstance(param, dict):
        raise TypeError
    if not isinstance(mode, str):
        raise TypeError
    if not isinstance(target, pd.DataFrame):
        raise TypeError

    scaler = np.array(param.get('scaler'))
    model = np.array(param.get('model'))
    bestDi = {}
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

    if mode == "score":
        print("---------------scoring start-------------------")
        for s in scaler:
            X_train_scale, X_test_scale = scaled(X_train, X_test, s)
            for m in model:
                temp, file_path = scoring(m, X_train_scale, X_test_scale, y_train, y_test,s,"Classifier")
                bestDi[file_path] = temp
                print(file_path, bestDi[file_path])
    elif mode == "fit":
        print("---------------fitting start-------------------")
        for s in scaler:
            X_train_scale, X_test_scale = scaled(X_train, X_test, s)
            for m in model:
                temp, file_path = predict(m, X_train_scale, X_test_scale, y_train, y_test,s,"Classifier")
                bestDi[file_path] = temp
                print(file_path, bestDi[file_path])

    return bestDi,file_path
    #return max(bestDi, key=bestDi.get), max(bestDi.values())

main.py
params = {
    "scaler": ["standard", "robust", "minmax"],
    "model": ["adaboost", "decisiontree", "bagging", "XGBoost", "gradient", "randomforest", "KNN"]
}

selected_params = {
    "scaler": ["standard", "robust", "minmax"],
    "model": ["decisiontree"]
}

# df[0] : original ,df[1] : oversampled, df[2] : droped outlier from original df[3] : droped outlier from oversampled

x,y = drop_data(df[0])
best_params, models = findbest.bestSearch("score", params, x, y)
print(best_params)
best_params = sorted(best_params.items(), key=lambda x: x[1], reverse=True)
best_params = dict(best_params[:5])
print(best_params)

print("End")
```

For code handling we functionalized the method, required for modeling, in the findbest.py file. We used modeling method “adaboost”, “decision tree”, “bagging”, “XGBoost”, “gradient”, “random forest” and “KNN”. 
The modeling function entered the scaler and model required by the factor value in the form of dict, and processed all models to be executed as a single method call. The bestSearch method returns the accuracy of the model and a single string of scaling and modeling methods. Using this return value, main.py finds the optimal modeling method by sorting the return value.

The following are the result values for each model.

- RESULT
![image](https://user-images.githubusercontent.com/84771856/171909185-53a46ef4-a6f7-4ac5-b390-7dca0597d953.png)


### 4. Best models

-	Original data set’s best 5 models
1.	XGBoost, Standard Scaling (0.9793)
2.	Random Forest, Standard Scaling (0.9792)
3.	Random Forest, Robust Scaling (0.9789)
4.	XGBoost, MinMax Scaling (0.9789)
5.	XGBoost, Robust Scaling (0.9787)

-	Oversampled data set’s best 5 models
1.	XGBoost, MinMax Scaling (0.9849)
2.	Random Forest, MinMaxScaling (0.9832)
3.	XGBoost, Robust Scaling (0.9817)
4.	Random Forest, Robust Scaling (0.981)
5.	XGBoost, Standard Scaling (0.9808)






