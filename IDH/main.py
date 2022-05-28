import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

df = pd.read_csv("../Dataset/star_classification.csv")

print("_____ Data type _____")
df.info()
print("_____ statistical data _____")
print(df.describe().T)
print("_____ Head data _____")
print(df.head())
print("_____ Count of dirty data _____")
print(df.isna().sum())
print("_____ Number of each target value _____")
print(df["class"].value_counts())

df['class'] = df['class'].replace({'GALAXY': 0,
                                   'STAR': 1,
                                   'QSO': 2})

sns.countplot(df["class"], palette="Set3")
plt.title("Class ", fontsize=10)
plt.show()


# Find outlier and delete outlier
shape1 = df.shape

for column in df.select_dtypes(include="number").columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    minimum = q1 - (1.5 * iqr)
    maximum = q3 + (1.5 * iqr)

    min_in = df[df[column] < minimum].index
    max_in = df[df[column] > maximum].index

    df.drop(min_in, inplace=True)
    df.drop(max_in, inplace=True)

shape2 = df.shape

outliers = shape1[0] - shape2[0]

print("\n\nTotal count of deleted outliers: ", outliers)


plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", linewidths=.7)
plt.show()

# Feature Selection
df = df.drop(['obj_ID', 'alpha', 'delta', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'fiber_ID'], axis=1)

# After dealing outlier
print("_____ After dealing outlier _____")
print(df.info())


x = df.drop('class', axis=1)
target = df['class']

bestParam = {
    "scaler": ["standard", "robust", "minmax"],
    "encoder": ["labelEncoder", "oneHotEncoder"],
    "model": ["LinearRegression","adaboost", "decisiontree", "bagging", "XGBoost", "gradient", "randomforest"]
}

# best_params, best_score = findBest.bestSearchEncoding(bestParam, x, target)
# print ("Best Combination, Score:", best_params, best_score)
