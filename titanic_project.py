# -*- coding: utf-8 -*-
# %% [markdown]

"""
Homework:

The folder '~//data//homework' contains data of Titanic with various features and survivals.

Try to use what you have learnt today to predict whether the passenger shall survive or not.

Evaluate your model.
"""
# %%
# load data
import pandas as pd

data = pd.read_csv(r"C:\Users\jl152\Desktop\train.csv")
df = data.copy()
df.sample(10)
# %%
# delete some features that are not useful for prediction
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
df.info()
# %%
# check if there is any NaN in the dataset
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))

# Handle missing values
df = df.copy()  # Create an explicit copy to avoid chain assignment
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Cabin'] = df['Cabin'].fillna('Unknown')

print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
# %%
# Extract cabin letter as a new feature
df['Cabin_Letter'] = df['Cabin'].str.extract('([A-Z])')
df.drop(columns=['Cabin'], inplace=True)

# Convert categorical data into numerical data using one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Cabin_Letter'])
df.sample(10)
# %%
# separate the features and labels
X = df.drop(columns=['Survived'])
y = df['Survived']
# %%
# train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# build model
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
# %%
# predict and evaluate
from sklearn.metrics import accuracy_score, classification_report

# Predict and evaluate SVM
svm_pred = svm_model.predict(X_test)
print("SVM Accuracy: ", accuracy_score(y_test, svm_pred))
print("SVM Classification Report:\n", classification_report(y_test, svm_pred))

# Predict and evaluate KNN
knn_pred = knn_model.predict(X_test)
print("KNN Accuracy: ", accuracy_score(y_test, knn_pred))
print("KNN Classification Report:\n", classification_report(y_test, knn_pred))

# Predict and evaluate Random Forest
rf_pred = rf_model.predict(X_test)
print("Random Forest Accuracy: ", accuracy_score(y_test, rf_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))