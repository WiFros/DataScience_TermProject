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
import findBest_2

df = pd.read_csv("star_classification.csv")
df.head()
df.info()

df["class"].value_counts()
sns.countplot(df["class"], palette="Set3")
plt.title("Class ",fontsize=10)
plt.show()

f,ax = plt.subplots(figsize=(12,8))
sns.heatmap(df.corr(), cmap="PuBu", annot=True, linewidths=0.5, fmt= '.2f',ax=ax)
plt.show()

le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])
df["class"] = df["class"].astype(int)

df = df.drop(['obj_ID','alpha','delta','run_ID','rerun_ID','cam_col','field_ID','fiber_ID'], axis = 1)

x = df.drop('class', axis=1)
y = df['class']

bestParam = {
        "scaler": ["standard", "robust", "minmax"],
        "encoder": ["labelEncoder", "oneHotEncoder"],
        "model": ["LinearRegression","adaboost", "decisiontree", "bagging", "XGBoost", "gradient", "randomforest"]
}
# best_params, best_score = findBest_2.bestSearchEncoding(bestParam, x, y)
# print ("Best Combination, Score:", best_params, best_score)

best_params, best_score = findBest_2.bestSearch(bestParam, x, y)
print ("Best Combination, Score:", best_params, best_score)
print("End")
