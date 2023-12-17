# -*- coding: utf-8 -*-
"""
10/11/2020
Frédéric Malharin

Analysis of a dataset related to SARS-COV2 with Python in Jupyter Notebooks and Spyder
The goal is to predict the result of the boolean variable "SARS-Cov-2 exam result"

This dataset comes from the course Machine Learnia (https://machinelearnia.com)
Many thanks to Guillaume Saint-Cirgue

SECOND PART : PREPROCESSING

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures

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

# model = DecisionTreeClassifier(random_state=0)
# model = RandomForestClassifier(random_state=0)

k_neighbours = 4

# model = make_pipeline(SelectKBest(f_classif, k=k_neighbours),
#                       DecisionTreeClassifier(random_state=0))

model = make_pipeline(
                      # PolynomialFeatures(2),
                      SelectKBest(f_classif, k=k_neighbours),
                      RandomForestClassifier(random_state=0)
                      )


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

evaluation(model)

#------------------------------------------------------FEATURE IMPORTANCE

# feat_imp = pd.DataFrame(model.feature_importances_, index=X_train.columns)
# print(feat_imp)
# feat_imp.plot.bar(figsize=(12,8))

# We can discard many variables, as only a few have some importance. Especially the viral variables.







