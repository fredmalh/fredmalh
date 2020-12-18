# -*- coding: utf-8 -*-
"""
18/12/2020
Analysis of a dataset related to wine quality.
With Python 3.8 (IDE : Spyder)
The goal is to predict the result of the quality value (integers from 1 to 10) based on the chemicals properties.

This dataset comes from UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/Wine+Quality

PART 3 : Optitmisation of a Decision Tree Classifier model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import *
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv(r'C:\Users\Fred\Dropbox\DATA SCIENCE\PORTFOLIO\wine-dataset\white_wine\white_wine.csv', sep = ';', error_bad_lines = False)
backup = df
pd.set_option('display.max_row', 20)
pd.set_option('display.max_column', 12)

# =================================================================================
#--------------------------------- --PART 3: OPTIMISATION OF DECISIONTREECLASSIFIER
# =================================================================================

print('-------------------------------------------------------------PART 3 : OPTIMISATION OF DECISIONTREECLASSIFIER')

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


#------------------------------------------------------TRAIN TEST SPLIT
print('-------------------------------------------------------------TRAIN TEST SPLIT')

trainset, testset = train_test_split(df, test_size=0.2)

print("trainset length: ", len(trainset), " // testset length: ", len(testset))

X_train = trainset.drop('quality', axis=1)
y_train = trainset['quality']
X_test = testset.drop('quality', axis=1)
y_test = testset['quality']
y_pred = []


#------------------------------------------------------MODEL
print('-------------------------------------------------------------MODEL')

DT_base_model = DecisionTreeClassifier()


#------------------------------------------------------VALIDATION CURVE
print('-------------------------------------------------------------VALIDATION CURVE')


# max_depth = range(1,15)
# min_samples_split = range(0, 200, 10)
# min_samples_leaf = range(5, 100, 5)
# max_leaf_nodes = np.arange(20, 1500, 20)

# train_score, val_score = validation_curve(DT_base_model, X_train, y_train, 'max_leaf_nodes', max_leaf_nodes, cv=5)

# plt.figure(figsize=(12,8))
# plt.plot(max_leaf_nodes, train_score.mean(axis=1), label='train_score')
# plt.plot(max_leaf_nodes, val_score.mean(axis=1), label='val_score')
# plt.ylabel('score')
# plt.xlabel('max_leaf_nodes')
# plt.ylim([0.5, 1.0])
# plt.legend()
# plt.show()

"""
The model is overfit, with all 4 hyperparameters.
I can add a model where I pick the hyperparameters to reduce overfitting, but it should also reduce the scores.
"""

DT_underfit_model = DecisionTreeClassifier(max_depth=6,
                                           min_samples_split=50,
                                           min_samples_leaf=50,
                                           max_leaf_nodes=50)

    
#------------------------------------------------------BASE MODEL VS RANDOMIZED SEARCH CV
print('-------------------------------------------------------------BASE MODEL VS RANDOMIZED SEARCH CV')


DT_metrics_matrix = pd.DataFrame(index = ['DT_base_model', 'DT_underfit_model', 'DT_random_best_model', 'DT_grid_best_model'],
                                 columns = ['Train_score',
                                            'Test_score',
                                            'MAE (absolute values)',
                                            'MAPE (%)',
                                            'Correct predictions (%)',
                                            'Predictions under 1 unit error (%)',
                                            'Maximum error (quality units)'])

DT_random_params = {
                    'max_depth': [1,2,4,6,8,10,15,20],
                    'min_samples_split': [1,2,5,10,25,50,100,200],
                    'min_samples_leaf': [1,2,5,10,25,50,100,200],
                    'max_leaf_nodes': [10,25,50,100,500,1000,2000],
                    }


random = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=DT_random_params, cv=4, n_iter=100)
random.fit(X_train, y_train)
DT_random_best_params = random.best_params_
DT_random_best_model = random.best_estimator_
print('Random best parameters :', DT_random_best_params)


evaluation(DT_base_model)
DT_metrics_matrix.iloc[0] = scoring_metrics

evaluation(DT_underfit_model)
DT_metrics_matrix.iloc[1] = scoring_metrics

evaluation(DT_random_best_model)
DT_metrics_matrix.iloc[2] = scoring_metrics

print(DT_metrics_matrix)


#------------------------------------------------------BASE MODEL VS GRID SEARCH CV
print('-------------------------------------------------------------BASE MODEL VS GRID SEARCH CV')

"""
I tried to go for values that were not the best val_score but less overfit, but the model was still overfit and the test results got worse.
So I focus on values that optimize the test results.
"""

DT_grid_params = {'max_depth': [20,30],
                    'min_samples_split': [2,3],
                    'min_samples_leaf': [1,2],
                    'max_leaf_nodes': [100,500,1000,2000],
                    }

grid = GridSearchCV(DecisionTreeClassifier(), param_grid=DT_grid_params, cv=4)
grid.fit(X_train, y_train)
DT_grid_best_params = grid.best_params_
DT_grid_best_model = grid.best_estimator_
print('Grid best parameters :', DT_grid_best_params)

evaluation(DT_grid_best_model)
DT_metrics_matrix.iloc[3] = scoring_metrics

print(DT_metrics_matrix)

"""
The best  and distribution are the ones with optimisation with GridSearchCV, though very overfit.
I could reduce the overfitting, but the results are a bit lower, and the distribution has a different shape.

The prediction results with GridSearchCV, despite the overfitting, are not that bad.
60% of the predictions are accurate.
Another 32% are mistaken by only 1 quality unit.
Only 8% of the predictions are mistaken by more than 2 quality units.
"""

