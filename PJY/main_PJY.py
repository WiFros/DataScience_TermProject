import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder, \
    OrdinalEncoder
from sklearn.cluster import KMeans
import numpy as np
from imblearn.over_sampling import SMOTE

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

bestParam = {
    "scaler": ["standard", "robust", "minmax"],
    "model": ["KNN", "adaboost", "decisiontree", "bagging", "XGBoost", "gradient", "randomforest","kmean"]
}

best_params, best_score = findBest_2.bestSearch(bestParam, x, y)
print ("Best Combination, Score:", best_params, best_score)
print("End")
