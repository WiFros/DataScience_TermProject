import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot");
sns.set(rc={'figure.figsize': (12, 7)})

# Encoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

# Test_train split
from sklearn.model_selection import train_test_split

# Scaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# Algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold as skf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV



# exploratory data analysis
df = pd.read_csv("../star_classification.csv")
print(df.head())
print(df.tail())
print(df.info())
print(df.isnull().sum())
print(df["class"].value_counts())
print(df.describe())

sns.countplot(df["class"], palette="Set3")
plt.title("Class ", fontsize=10)
plt.show()
df["class"] = [0 if i == "GALAXY" else 1 if i == "STAR" else 2 for i in df["class"]]

f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="PuBu", annot=True, linewidths=0.5, fmt='.2f', ax=ax)
plt.show()

corr = df.corr()
corr["class"].sort_values()