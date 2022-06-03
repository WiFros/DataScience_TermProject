import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder

import numpy as np
import findbest


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


#MAIN
df = []
df.append(pd.read_csv("star_classification.csv"))
# OverSampling
# --------------------------------------------------
under_data = UnderSampling(df[0], 'GALAXY', 40000)
over_sampler = SMOTE()
X = under_data.drop(columns='class')
y = under_data['class']

X_over, y_over = over_sampler.fit_resample(X, y)

over_data = pd.concat([X_over, y_over], axis=1)
over_data.to_csv("sampled_data.csv", index=False)
df.append(pd.read_csv("sampled_data.csv"))

# --------------------------------------------------


data_exploration(df[0],'original data','show')
data_exploration(df[1],'oversampled data','show')
df.append(rem_outliers(df[0]),0.25)
data_exploration(df[2],'draped outlier-original','show')
df.append(rem_outliers(df[1]),0.25)
data_exploration(df[3],'draped outlier-oversample','show')

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
