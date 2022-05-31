import operator

import joblib
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder, \
    OrdinalEncoder
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ROCAUC
import findBest_2



def UnderSampling(data, class_name, num_sample):
    if not isinstance(data, pd.DataFrame):
        raise TypeError

    div_data = data[data['class'] == class_name]
    left_data = data[data['class'] != class_name]

    div_data = div_data.sample(num_sample)
    sampled_data = pd.concat([div_data, left_data])

    return sampled_data.reset_index(drop=True)


df = pd.read_csv('star_classification.csv')

# OverSampling
# --------------------------------------------------
under_data = UnderSampling(df, 'GALAXY', 40000)

over_sampler = SMOTE()
X = under_data.drop(columns='class')
y = under_data['class']

X_over, y_over = over_sampler.fit_resample(X, y)

over_data = pd.concat([X_over, y_over], axis=1)
over_data.to_csv("sampled_data.csv", index=False)
# --------------------------------------------------
df_over = pd.read_csv("sampled_data.csv")
print(df.head())
print(df.info())

df["class"].value_counts()
sns.countplot(df["class"], palette="Set3")
plt.title("Class ", fontsize=10)
plt.show()

df_over["class"].value_counts()
sns.countplot(df_over["class"], palette="Set3")
plt.title("Over_Class ", fontsize=10)
plt.show()

f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="PuBu", annot=True, linewidths=0.5, fmt='.2f', ax=ax)
plt.show()

le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])
df["class"] = df["class"].astype(int)

df = df.drop(['obj_ID', 'alpha', 'delta', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'fiber_ID'], axis=1)

x = df.drop('class', axis=1)
y = df['class']

le_over = LabelEncoder()
df_over["class"] = le_over.fit_transform(df_over["class"])
df_over["class"] = df_over["class"].astype(int)
df_over = df_over.drop(['obj_ID', 'alpha', 'delta', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'fiber_ID'], axis=1)
x_over = df_over.drop("class", axis=1)
y_over = df_over["class"]

bestParam = {
    "scaler": ["standard", "robust", "minmax"],
    "model": ["adaboost", "decisiontree", "bagging", "XGBoost", "gradient", "randomforest", "KNN"]
}

selectedParam = {
    "scaler": ["standard", "robust", "minmax"],
    "model": ["decisiontree"]
}

best_params,models= findBest_2.bestSearch("scorez",bestParam, x_over, y_over)

print (best_params)
best_params = sorted(best_params.items(),key=lambda x:x[1],reverse=True)
best_params = dict(best_params[:5])
print(best_params)


print("End")
