# -*- coding: utf-8 -*-
"""
21/11/2020
Frédéric Malharin

Analysis of a dataset related to wine quality.
With Python 3.8 (IDE : Jupyter Notebooks and Spyder)
The goal is to predict the result of the quality value (integers from 1 to 10) based on the chemicals properties.

This dataset comes from UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/Wine+Quality


PART 1 : EXPLORATORY DATA ANALYSIS

PART 2 : MODEL SELECTION

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\Fred\Dropbox\DATA SCIENCE\EXERCICES\wine-dataset\white_wine\white_wine.csv', sep = ';', error_bad_lines = False)
backup = df


# ===============================================================================
#----------------------------------------------PART 1 : EXPLORATORY DATA ANALYSIS
# ===============================================================================

print('-------------------------------------------------------------PART 1 : EXPLORATORY DATA ANALYSIS')

pd.set_option('display.max_row', 20)
pd.set_option('display.max_column', 12)
df.head()
df.shape
df["quality"].unique()
df.dtypes.value_counts()
df.isna().sum()

"""
COMMENTS:
shape = 4898 rows, 12 columns
dtypes: 9 floats, 3 integers
Our target is the feature 'quality' (integer)
There is no Nan in this dataset.
"""

#------------------------------------------------------TARGET ANALYSIS
print('-------------------------------------------------------------TARGET EXPLORATION')

target_categories_frequencies = df["quality"].value_counts(normalize=True, sort=True)
round(target_categories_frequencies*100, 2)

"""
COMMENTS:
There is only 7 categories for our target feature 'quality'.
"""

#------------------------------------------------------VARIABLES ANALYSIS
print('-------------------------------------------------------------VARIABLES ANALYSIS')

for column in df:
    print(f'{column :-<30} {len(set(df[column]))} different values, with standard deviation = {round(df[column].std(), 3)}')

# for column in df:
#     plt.figure()
#     sns.distplot(df[column])

# sns.pairplot(df)
# df.corr()

sns.heatmap(df.corr())


"""
COMMENTS:
The highest correlation coefficient between the target and the other variables is 0.436 ('alcohol').
The other cofficients between the target and the other variables are lower than 0.1.
"""


#------------------------------------------------------PREPROCESSING
print('-------------------------------------------------------------PREPROCESSING')

"""
Preprocessing is not needed at this point, as there are non nans, the variables are numbers only.
I'll see later if I need to standardize or to create new features.
"""

# ===============================================================================
#--------------------------------------------------------PART 2 : MODEL SELECTION
# ===============================================================================

print('-------------------------------------------------------------PART 2 : MODEL SELECTION')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

"""
The target has 7 classes. Let's compare several classification models.
"""

DT = DecisionTreeClassifier()
RF = RandomForestClassifier()
GBC = GradientBoostingClassifier()
ADA = AdaBoostClassifier()
SGDC = SGDClassifier()
KNC = make_pipeline(StandardScaler(), KNeighborsClassifier())
SVC = make_pipeline(StandardScaler(), SVC())
LSVC = make_pipeline(StandardScaler(), LinearSVC())

dict_of_models = {'DT': DT,
                  'RF': RF,
                  'GBC': GBC,
                  'ADA': ADA,
                  'SGDC': SGDC,
                  'KNC': KNC,
                  'SVC': SVC,
                  'LSVC': LSVC,
                  }


#------------------------------------------------------TRAIN TEST SPLIT
print('-------------------------------------------------------------TRAIN TEST SPLIT')

trainset, testset = train_test_split(df, test_size=0.2)

print("trainset length: ", len(trainset), " // testset length: ", len(testset))



X_train = trainset.drop('quality', axis=1)
y_train = trainset['quality']
X_test = testset.drop('quality', axis=1)
y_test = testset['quality']
y_pred = []


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

def cross_val(model, X_train, y_train):
    score = cross_val_score(model, X_train, y_train, cv=5).mean
    return score

scoring_metrics = []

comparison_matrix = pd.DataFrame(index = dict_of_models.keys(), columns = ['Train_score',
                                                                           'Test_score',                                                                           
                                                                           'MAE (absolute values in quality units)',
                                                                           'MAPE (%)',
                                                                           'Correct predictions (%)',
                                                                           'Predictions under 1 unit error (%)',
                                                                           'Maximum error (quality units)'])


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
    n, bins, patches = plt.hist(error_hist, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.xlabel('error (quality units)')
    plt.ylabel('occurrences')
    plt.show()
    
def distribution(model):
    plt.figure(figsize=(12,8))
    plt.title(model)
    sns.distplot(y_test, label='y_test', kde=False, rug=False, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], color='darkgray')
    sns.distplot(y_pred, label='y_pred', kde=False, rug=False, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], color='gold')
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


#------------------------------------------------------MODEL COMPARISON AND SELECTION
print('-------------------------------------------------------------MODEL COMPARISON AND SELECTION')


for name, model in dict_of_models.items():
    evaluation(model)
    comparison_matrix.loc[name]= scoring_metrics

print(comparison_matrix)

"""
RandomForestClassifier shows the best scores, and is significantly overfit.
DecisionTreeClassifier shows the second best scores, and is also significantly overfit, but the distribution of the predictions is closer to the original distribution.
Among the less overfit models, SVC shows the best scores, but the distribution has a different shape.

I will optimise DecisionTreeClassifier.
"""

