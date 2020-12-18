# -*- coding: utf-8 -*-
"""
18/12/2020
Analysis of a dataset related to wine quality.
With Python 3.8 (IDE : Jupyter Notebooks and Spyder)
The goal is to predict the result of the quality value (integers from 1 to 10) based on the chemicals properties.

This dataset comes from UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/Wine+Quality

PART 4 : DIMENSIONALITY REDUCTION
Let's try to reduce dimensionality in order to reduce overfitting of the DecisionTree model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv(r'C:\Users\Fred\Dropbox\DATA SCIENCE\PORTFOLIO\wine-dataset\white_wine.csv', sep = ';', error_bad_lines = False)
backup = df
pd.set_option('display.max_row', 20)
pd.set_option('display.max_column', 12)

# =================================================================================
#--------------------------------------------------PART 4: DIMENSIONALITY REDUCTION
# =================================================================================

print('-------------------------------------------------------------PART 3 : OPTIMISATION OF RANDOMFORESTCLASSIFIER')

#------------------------------------------------------CUSTOM SCORING
print('-------------------------------------------------------------CUSTOM SCORING')

def correct_predictions(y_test, y_pred):
    error = 100*(sum((abs(y_pred-y_test)) == 0) / len(y_test))
    return round(error, 1)

def errors_under_1_quality_unit(y_test, y_pred):
    error = 100*(sum((abs(y_pred-y_test)) <= 1) / len(y_test))
    return round(error, 1)

def max_error(y_test, y_pred):
    error = max(np.abs(y_pred-y_test))
    return round(error, 0)

scoring_metrics = []


#------------------------------------------------------EVALUATION PROCESS
print('-------------------------------------------------------------EVALUATION PROCESS')

def model_fit_predict(model):
    global y_pred
    y_pred = []
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


def scoring(model):
    global y_pred
    global scoring_metrics
    scoring_metrics = []
    
    train_score = round(model.score(X_train, y_train), 3)
    test_score = round(model.score(X_test, y_test), 3)
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    mape = round(100 * mae/np.mean(y_test), 2)
    predictions_correct = correct_predictions(y_test, y_pred)
    predictions_errors_under_1_quality_unit = errors_under_1_quality_unit(y_test, y_pred)
    maximum_error = max_error(y_test, y_pred)

    scoring_metrics = [train_score,
                      test_score,
                      mae,
                      mape,
                      predictions_correct,
                      predictions_errors_under_1_quality_unit,
                      maximum_error]
    
def errors_fig(model):
    plt.figure(figsize=(12,8))
    plt.title(model)
    error_hist = abs(y_pred-y_test)
    n, bins, patches = plt.hist(error_hist, [0,1,2,3,4,5,6,7,8,9])
    plt.xlabel('error (quality units)')
    plt.ylabel('occurrences')
    plt.show()

def distribution(model):
    plt.figure(figsize=(12,8))
    plt.title(model)
    sns.distplot(y_test, label='y_test', kde=False, rug=False, bins=[0,1,2,3,4,5,6,7,8,9], color='darkgray')
    sns.distplot(y_pred, label='y_pred', kde=False, rug=False, bins=[0,1,2,3,4,5,6,7,8,9], color='gold')
    plt.xlabel('quality')
    plt.ylabel('occurrences')
    plt.legend()

def confusion_classification():
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def evaluation(model):
    model_fit_predict(model)
    scoring(model)
    errors_fig(model)
    distribution(model)
    confusion_classification()


#------------------------------------------------------METRICS TABLE
print('-------------------------------------------------------------METRICS TABLE')

DT_metrics_matrix = pd.DataFrame(columns = ['Train_score',
                                            'Test_score',
                                            'MAE (absolute values)',
                                            'MAPE (%)',
                                            'Correct predictions (%)',
                                            'Predictions under 1 unit error (%)',
                                            'Maximum error (quality units)'])


#------------------------------------------------------MODEL
print('-------------------------------------------------------------MODEL')

DT_base_model = DecisionTreeClassifier()

# #------------------------------------------------------FEATURE SELECTION
print('-------------------------------------------------------------FEATURE SELECTION')

# Define the features for X and y
X = df.drop('quality', axis=1)
y = df['quality']

# Select k best features
"""
I also tried with chi2 and mutual_info_classif, and after comparing the results and the features that were selected,
I decided to go on with f_classif
"""
k_list = range(2,12)

for k in k_list:
    selector = SelectKBest(f_classif, k=k)
    selector.fit_transform(X,y)
    columns_filter_f_classif = selector.get_support()

    X_columns = np.array(X.columns.values)
    selected_features_columns = np.where(columns_filter_f_classif)
    selected_features = X_columns[selected_features_columns]
    df = df[selected_features]
    df['quality'] = y
    
    trainset, testset = train_test_split(df, test_size=0.2)
    X_train = trainset.drop('quality', axis=1)
    y_train = trainset['quality']
    X_test = testset.drop('quality', axis=1)
    y_test = testset['quality']
    y_pred = []
    
    evaluation(DT_base_model)
    DT_metrics_matrix.loc[f'{k}'] = scoring_metrics
    
    df = backup
    

print(DT_metrics_matrix)

    
plt.figure(figsize=(12,8))
plt.plot(DT_metrics_matrix.index.values, DT_metrics_matrix['Train_score'], label='train_score')
plt.plot(DT_metrics_matrix.index.values, DT_metrics_matrix['Test_score'], label='test_score')
plt.plot(DT_metrics_matrix.index.values, DT_metrics_matrix['Predictions under 1 unit error (%)']/100, label='under_1_unit_error')
plt.ylabel('score')
plt.xlabel('k')
# plt.ylim([0.5, 1.0])
plt.legend()
plt.show()   

"""
Between 4 and 11 features, there are no significant differences in the scores and overfitting.
With 2 or 3 variables, the results are bad even for the train set.

"""
