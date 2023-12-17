# -*- coding: utf-8 -*-
"""
20/11/2020
Frédéric Malharin

Analysis of a dataset related to SARS-COV2 with Python in Jupyter Notebooks and Spyder
The goal is to predict the result of the boolean variable "SARS-Cov-2 exam result"

This dataset comes from the course Machine Learnia (https://machinelearnia.com)
Many thanks to Guillaume Saint-Cirgue

THIRD PART : MODEL

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import precision_recall_curve, recall_score


#------------------------------------------------------RESET THE DATASET

df = pd.read_excel(r'C:\Users\PcCom\Dropbox\DATA SCIENCE\PORTFOLIO\SARS-COV2-dataset\dataset.xlsx')
backup = df.copy()
pd.set_option('display.max_row', 111)


#------------------------------------------------------CODING

code = {'negative':0,
        'positive':1,
        'not_detected':0,
        'detected':1}
for column in df.select_dtypes('object').columns:
    df.loc[:, column] = df[column].map(code)


#------------------------------------------------------CREATE SUB SETS

missing_rate = df.isna().sum()/df.shape[0]
blood_columns = df.columns[(missing_rate < 0.9) & (missing_rate > 0.88)]
blood_columns = list(blood_columns)
viral_columns = df.columns[(missing_rate < 0.88) & (missing_rate > 0.75)]
viral_columns = list(viral_columns)
viral_columns.remove('Influenza B, rapid test')
viral_columns.remove('Influenza A, rapid test')


#------------------------------------------------------FEATURE ENGINEERING

df['is_sick'] = df[viral_columns].sum(axis=1) >= 1

key_columns = ['Patient age quantile', 'SARS-Cov-2 exam result', 'is_sick']
df = df[key_columns + blood_columns]


#------------------------------------------------------TRAIN SET AND TEST SET

trainset, testset = train_test_split(df, test_size=0.2, random_state=0)
trainset['SARS-Cov-2 exam result'].value_counts()
testset['SARS-Cov-2 exam result'].value_counts()


#------------------------------------------------------PREPROCESSING FUNCTION

def cleaning_nan(data_frame):
    return data_frame.dropna(axis=0)

def preprocessing(data_frame):
    data_frame = cleaning_nan(data_frame)
    X = data_frame.drop('SARS-Cov-2 exam result', axis=1)
    y = data_frame['SARS-Cov-2 exam result']
    print(y.value_counts())
    return X, y

X_train, y_train = preprocessing(trainset)
X_test, y_test = preprocessing(testset)


#------------------------------------------------------MODEL

preprocessor = make_pipeline(
                              PolynomialFeatures(2, include_bias=False),
                             SelectKBest(f_classif, k=10)
                             )

# RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
# AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
SVM = make_pipeline(preprocessor, StandardScaler(), SVC(random_state=0))
# KNN = make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())

# dict_of_models = {'RandomForest':RandomForest, 'AdaBoost':AdaBoost, 'SVM':SVM, 'KNN':KNN}

#------------------------------------------------------EVALUATION

def evaluation(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                               cv=4, scoring='f1',
                                               train_sizes=np.linspace(0.1,1,10))
    
    plt.figure(figsize=(12,8))
    plt.plot(N, train_score.mean(axis=1), label='train_score')
    plt.plot(N, val_score.mean(axis=1), label='validation_score')
    plt.legend()


# evaluation(SVM)


#------------------------------------------------------COMPARISON OF THE MODELS

# for name, model in dict_of_models.items():
#     print(name)
#     evaluation(model)

"""
We keep SVM, no overfitting, recall is promising
"""


#------------------------------------------------------OPTIMISATION OF THE SVC MODEL WITH GRID SEARCH CV

# hyper_params = {
#                 'svc__gamma':[1e-1, 1e-2, 1e-3, 1e-4],
#                 'svc__C':[1, 10, 100, 500, 1000, 2000, 5000]
#                 }

# grid = GridSearchCV(SVM, hyper_params, scoring='recall', cv=4)
# grid.fit(X_train, y_train)
# print(grid.best_params_)
# y_pred = grid.predict(X_test)
# print(classification_report(y_test, y_pred))

# evaluation(grid.best_estimator_)

"""
We change to RandomizedSearchCV to change more hyper parameters, otherwise it would take too much processing time.
"""


#------------------------------------------------------OPTIMISATION OF THE SVC MODEL WITH RANDOM SEARCH CV

hyper_params = {'svc__gamma':[1e2, 1e-3, 1e-4],
                'svc__C':[1,10,100, 1000],
                'pipeline__polynomialfeatures__degree':[2, 3, 4],
                'pipeline__selectkbest__k': range(4, 100)
                }

grid = RandomizedSearchCV(SVM, hyper_params, scoring='recall', cv=4, n_iter=40)
grid.fit(X_train, y_train)
print(grid.best_params_)
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))

evaluation(grid.best_estimator_)

"""
Best parameters :
{'svc__gamma': 0.001, 'svc__C': 1100, 'pipeline__selectkbest__k': 21, 'pipeline__polynomialfeatures__degree': 3}
"""

#------------------------------------------------------PRECISION RECALL CURVE

print(grid.best_estimator_.decision_function)
# Out[23]: <function sklearn.pipeline.Pipeline.decision_function(self, X)>

precision, recall, threshold = precision_recall_curve(y_test, grid.best_estimator_.decision_function(X_test))
plt.figure()
plt.plot(threshold, precision[:-1], label='precision')
plt.plot(threshold, recall[:-1], label='recall')
plt.legend()

def final_model(model, X, threshold=0):
    return model.decision_function(X) > threshold

y_pred = final_model(grid.best_estimator_, X_test, threshold=1)
print(f1_score(y_test, y_pred))
print(recall_score(y_test, y_pred))

"""
I can't get a better recall better than 0.50 for the prediction of a person testing positive to SARS-Cov-2.
The prediction is not acceptable.
"""

