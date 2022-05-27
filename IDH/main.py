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
import findBest

df = pd.read_csv("../Dataset/star_classification.csv")

le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])
df["class"] = df["class"].astype(int)
df = df.drop(["obj_ID"], axis=1)
df = df.drop("rerun_ID", axis=1)
df = df.drop("run_ID", axis=1)
df = df.drop("field_ID", axis=1)
df = df.drop("fiber_ID", axis=1)
df = df.drop("cam_col", axis=1)

x = df.drop('class', axis=1)
target = df['class']

bestParam = {
        "scaler": ["standard", "robust", "minmax"],
        "encoder": ["labelEncoder", "oneHotEncoder"],
        "model": ["adaboost", "decisiontree", "bagging", "XGBoost", "gradient", "randomforest"]
}

best_params, best_score = findBest.bestSearchEncoding(bestParam, x, target)
print ("Best Combination, Score:", best_params, best_score)