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
y = df['class']

#StandardScaler_RandomForestClassifier
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, shuffle=True)

scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)
RFC = RandomForestClassifier(random_state=42)
RFC.fit(x_train,y_train)
y_predicted = RFC.predict(x_test)
score = RFC.score(x_test, y_test)
rf_score_ = np.mean(score)
print('RandomForest Accuracy : %.3f' % (rf_score_))

# _____최적의 K값 찾기_____
# k에따른 accuracy 저장
# accuracy_list = []
#
# k_range = range(3,40)
# # 각 k마다 모델 테스트
# for k in k_range:
#     model = KNeighborsClassifier(n_neighbors=k)
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     accuracy_list.append(accuracy_score(y_test, y_pred))

# 시각화
# plt.plot(k_range, accuracy_list, 'o--', color='orange')
# plt.xlabel("k")
# plt.ylabel("test accuracy")
# plt.show()

# tmp = max(accuracy_list)
# index = accuracy_list.index(tmp)+3
# kNN 모델 선언
k = 3
model = KNeighborsClassifier(n_neighbors = k)
# 모델 학습
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('KNeighbors Accuracy : %.3f' % accuracy_score(y_test,y_pred))

#StratifiedKFold
skfold = StratifiedKFold(n_splits=3, shuffle = True, random_state=42)
scores = cross_val_score(model, x_train, y_train, cv=skfold)
print('StratifiedKFold Accuracy : %.3f' % np.mean(scores))

#KFold
kfold = KFold(n_splits=3, shuffle = True, random_state=42)
scores2 = cross_val_score(model, x_train, y_train, cv=kfold)
print('KFold Accuracy : %.3f' % np.mean(scores2))

#LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
print('LinearRegression Accuracy : %.3f' % model.score(x_test, y_test))
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

# #MinMaxScaler_RandomForestClassifier
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, shuffle=True)
#
# scalar = MinMaxScaler()
# x_train = scalar.fit_transform(x_train)
# x_test = scalar.transform(x_test)
#
# RFC = RandomForestClassifier(random_state=42)
# RFC.fit(x_train,y_train)
# y_predicted = RFC.predict(x_test)
# score = RFC.score(x_test, y_test)
# rf_score_ = np.mean(score)
# print('Accuracy : %.3f' % (rf_score_))
#
# #RobustScaler_RandomForestClassifier
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, shuffle=True)
#
# scalar = RobustScaler()
# x_train = scalar.fit_transform(x_train)
# x_test = scalar.transform(x_test)
#
# RFC = RandomForestClassifier(random_state=42)
# RFC.fit(x_train,y_train)
# y_predicted = RFC.predict(x_test)
# score = RFC.score(x_test, y_test)
# rf_score_ = np.mean(score)
# print('Accuracy : %.3f' % (rf_score_))