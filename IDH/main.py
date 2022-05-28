import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import findBest

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

sns.countplot(df["class"], palette="Set3")
plt.title("Class ", fontsize=10)
plt.show()



# Galaxies are easier to seperate than stars and stars could have done.
# Although quasars and stars shows parallel quantities, they have some distinctive statistical distributions to filters to help to decide which one it is.
# Galaxies는 직관적으로 분류가 되는데 stars와 QSO는 아니다. 그러므로 boxplot을 통해 둘을 비교
# for column in ['alpha', 'delta','r', 'i','field_ID','redshift', 'plate', 'MJD', 'fiber_ID']:
#     plt.figure(figsize=(12,6))
#     sns.histplot(data=df, x=column, kde=True, hue="class")
#     plt.title(column)
#     plt.show()



# alpha , delta , field_ID distributions are quite similar for stars and quasars.
# r i and redshift on the other hand, are distinctive features between these.
# Since there are a lot of outliers, I'll clean some of them.
# STAR와 QSO를 비교했을 때 위의 설명대로 몇개는 비슷한 성질을 띄고 있어서 필요없는 데이터이지만 몇개는 다른점을 띄어 필요한 데이터임 또한 outlier가 많이 존재함
star_and_qso = df[(df["class"] == "STAR") | (df["class"] == "QSO")]

# for column in ['alpha', 'delta','r', 'i','field_ID','redshift', 'plate', 'MJD', 'fiber_ID']:
#     plt.figure(figsize=(12,6))
#     plt.title(column)
#     sns.boxplot(data=star_and_qso, x=column, y="class", palette="flare", linewidth=1.5, saturation=0.6)
#     plt.show()

df = df.drop(['alpha', 'delta', 'field_ID'], axis=1)


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
df = df.drop(['obj_ID',  'run_ID', 'rerun_ID', 'cam_col', 'fiber_ID'], axis=1)

# After dealing outlier
print("_____ After dealing outlier _____")
print(df.info())

df['class'] = df['class'].replace({'GALAXY': 0,
                                   'STAR': 1,
                                   'QSO': 2})
x = df.drop('class', axis=1)
target = df['class']

param = {
    "scaler": ["standard", "robust", "minmax"],
    "model": ["KNN", "decisiontree", "bagging", "adaboost", "XGBoost", "gradient", "randomforest"]
}

best_params, best_score = findBest.bestSearch(param, x, target)
print("Best Combination, Score:", best_params, best_score)
