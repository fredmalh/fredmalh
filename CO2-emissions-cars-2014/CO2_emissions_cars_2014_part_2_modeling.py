# -*- coding: utf-8 -*-
"""
23/11/2020

Analysis of a dataset related to fuel consumption and CO2 emissions of cars in 2014.
With Python 3.8 (IDE : Spyder)
The goal is to analyse the relationships between the variables (especially the vehicle class)
and the CO2 consumption. I'll train ML models to try to predict CO2 emisions based on the other variables.

PART 2 - MODELING
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree


df = pd.read_csv(r"C:\...\FuelConsumptionCo2.csv")
backup = df

pd.set_option('display.max_row', 20)
pd.set_option('display.max_column', 13)

# =============================================================================
#------------------------------------------------------- PART 2 : PREPROCESSING
# =============================================================================

"""
I apply the same changes that I've done in the first part :
    - I drop the NaN
    - I only keep the columns that I want to use for the models
    - I code the categorical data
"""


print('-------------------------------------------------------------PART 2 : PREPROCESSING')

df = df.dropna(axis=0)
df = df[['VEHICLECLASS', 'ENGINESIZE', 'CYLINDERS', 'CO2EMISSIONS']]

code={}

for column in df.select_dtypes('object').columns:
    code[f"code_{column}"] = {}
    list1 = list(df[column].unique())
    i=0
    for val in list1:
        code[f"code_{column}"][f"{val}"] = i
        i+=1

for column in df.select_dtypes('object').columns:
    df.loc[:, column] = df[column].map(code[f"code_{column}"])

print(df)


# =============================================================================
#------------------------------------------------------------ PART 3 : MODELING
# =============================================================================

print('-------------------------------------------------------------PART 3 : MODELING')

"""
The target is continuous. Let's compare several regression models.
"""

Linear_Regression = LinearRegression()
Polynomial_2_Regression = make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression())
Polynomial_3_Regression = make_pipeline(PolynomialFeatures(3, include_bias=False), LinearRegression())
DecisionTree = DecisionTreeRegressor()
RandomForest = RandomForestRegressor()
SVM = make_pipeline(StandardScaler(), SVR())
KNN = make_pipeline(StandardScaler(), KNeighborsRegressor())
SGDR = SGDRegressor()
GBR = GradientBoostingRegressor()
ADA = AdaBoostRegressor()


dict_of_models = {'Linear_Regression': Linear_Regression,
                  'Polynomial_2_Regression': Polynomial_2_Regression,
                  'Polynomial_3_Regression': Polynomial_3_Regression,
                  'DecisionTreeRegressor': DecisionTree,
                  'RandomForestRegressor': RandomForest,
                  'SVM': SVM,
                  'KNN': KNN,
                  'SGDR': SGDR,
                  'GBR': GBR,
                  'ADA': ADA,
                  }


#------------------------------------------------------CUSTOM SCORING
print('-------------------------------------------------------------CUSTOM SCORING')

def under_10_percent_error(y_test, y_pred):
    error = 100*(sum(abs(y_pred-y_test)/y_test < 0.1) / len(y_test))
    return round(error, 2)

def max_error(y_test, y_pred):
    error = 100*max(abs(y_pred-y_test)/y_test)
    return round(error, 2)

comparison_matrix = pd.DataFrame(index = dict_of_models.keys(), columns = ['Train_score',
                                                                           'Test_score',
                                                                           'r2',
                                                                           'MAE (absolute values)',
                                                                           'MAE (%)',
                                                                           'Accuracy (%)',
                                                                           'Predictions under 10% error (%)',
                                                                           'Maximum error (%)'])



#------------------------------------------------------TRAIN TEST SPLIT
print('-------------------------------------------------------------TRAIN TEST SPLIT')

trainset, testset = train_test_split(df, test_size=0.2, random_state=0)

print("trainset length: ", len(trainset['CO2EMISSIONS']), " // testset length: ", len(testset['CO2EMISSIONS']))

def variable_selection(df):
    X = df[['VEHICLECLASS', 'ENGINESIZE', 'CYLINDERS']]
    y = df['CO2EMISSIONS']
    return X, y

X_train, y_train = variable_selection(trainset)
X_test, y_test = variable_selection(testset)
y_pred = []

#------------------------------------------------------EVALUATION PROCESS
print('-------------------------------------------------------------EVALUATION PROCESS')

def model_fit_predict():
    global y_pred
    y_pred = []
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

def scoring():
    global scoring_values
    scoring_values = []
    train_score = round(model.score(X_train, y_train), 3)
    test_score = round(model.score(X_test, y_test), 3)
    r2 = round(r2_score(y_test, y_pred), 3)
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    mape = round(100 * mae/np.mean(y_test), 2)
    accuracy = 100-mape
    predictions_under_10_percent_error = under_10_percent_error(y_test, y_pred)
    maximum_error = max_error(y_test, y_pred)
    
    scoring_values = [train_score,
                      test_score,
                      r2,
                      mae,
                      mape,
                      accuracy,
                      predictions_under_10_percent_error,
                      maximum_error]

def scatter():
    # I display the predictions with 1 dimension only, but it was trained with 3 dimensions.    
    plt.figure(figsize=(12,8))
    plt.title(model)
    plt.scatter(testset['ENGINESIZE'],  y_test, label='y_test', c='dodgerblue')
    plt.scatter(testset['ENGINESIZE'], y_pred, label='y_pred', c='darkorange')
    plt.legend()
    plt.show()

def errors_histogram():
    plt.figure(figsize=(12,8))
    plt.title(model)
    error_hist = np.abs(100*(y_test - y_pred)/y_test)
    plt.hist(error_hist, bins=50)
    plt.xlabel('% error')
    plt.ylabel('occurrences')
    plt.show()
    
def evaluation(model):
    model_fit_predict()
    scoring()
    scatter()
    errors_histogram()


#------------------------------------------------------MODEL COMPARISON AND SELECTION
print('-------------------------------------------------------------MODEL COMPARISON AND SELECTION')

scoring_values = []

for name, model in dict_of_models.items():
    evaluation(model)
    comparison_matrix.loc[name]= scoring_values

print(comparison_matrix)

"""
The histograms show well the outliers, we can also see them on the plot.
Those will be impossible to predict. I'll just ignore them.

RandomForest, DecisionTree and GBR seem to be the best models.
For the same result, DecisionTree is simpler and requires less ressources than RandomForest.
I'll start with DecisionTree, and then I'll try GBR.
"""


# =================================================================================
#------------------------------------PART 4 : OPTIMISATION OF DECISIONTREEREGRESSOR
# =================================================================================

print('-------------------------------------------------------------PART 4 : OPTIMISATION OF DECISIONTREEREGRESSOR')

DT_base_model_1 = DecisionTreeRegressor(random_state=3)
DT_base_model_2 = DecisionTreeRegressor(random_state=4)
DT_base_model_3 = DecisionTreeRegressor(random_state=5)


DT_metrics_matrix = pd.DataFrame(index = ['DT_base_model_1', 'DT_base_model_2', 'DT_base_model_3', 'DT_random_best_model', 'DT_grid_best_model'],
                                  columns = ['Train_score',
                                            'Test_score',
                                            'r2',
                                            'MAE (absolute values)',
                                            'MAE (%)',
                                            'Accuracy (%)',
                                            'Predictions under 10% error (%)',
                                            'Maximum error (%)'])

#------------------------------------------------------TRAIN TEST SPLIT
print('-------------------------------------------------------------TRAIN TEST SPLIT')

trainset, testset = train_test_split(df, test_size=0.2, random_state=1)

X_train, y_train = variable_selection(trainset)
X_test, y_test = variable_selection(testset)
y_pred = y_test

#------------------------------------------------------EVALUATION FUNCTION
print('-------------------------------------------------------------EVALUATION FUNCTION')

def search_cv_evaluation(model):
    model_fit_predict()
    scoring()
    
    
#------------------------------------------------------BASE MODEL VS RANDOMIZED SEARCH CV
print('-------------------------------------------------------------BASE MODEL VS RANDOMIZED SEARCH CV')

DT_random_params = {'max_depth': [2, 3, 5, 10, 15, 20, 50, None],
                    'min_samples_split': range(2, 10),
                    'min_samples_leaf': range(1, 5),
                    'max_leaf_nodes': [5, 10, 20, 50, 100, None],
                    'min_impurity_decrease':[0.00001, 0.00005, 0.0001, 0.001, 0.01, 0.1, 0.5],
                    'ccp_alpha':[0.00001, 0.00005, 0.0001, 0.001, 0.01, 0.1, 0.5]
                    }

random = RandomizedSearchCV(DecisionTreeRegressor(), param_distributions=DT_random_params, cv=4, n_iter=100)
random.fit(X_train, y_train)
DT_random_best_params = random.best_params_
DT_random_best_model = random.best_estimator_
print('Random best parameters :', DT_random_best_params)

# search_cv_evaluation(DT_base_model_1)
# DT_metrics_matrix.iloc[0] = scoring_values

# search_cv_evaluation(DT_base_model_2)
# DT_metrics_matrix.iloc[1] = scoring_values

# search_cv_evaluation(DT_base_model_3)
# DT_metrics_matrix.iloc[2] = scoring_values

# print(DT_metrics_matrix)

"""
There is no improvement whatsoever. Let's see with GRID SEARCH CV.
"""

#------------------------------------------------------BASE MODEL VS GRID SEARCH CV
print('-------------------------------------------------------------BASE MODEL VS GRID SEARCH CV')

DT_grid_params = {'max_depth': [15],
                  'min_samples_split': [2],
                  'min_samples_leaf': [1,2,3],
                  'max_leaf_nodes': [40, 50, 60],
                  'min_impurity_decrease':[5e-05, 1e-04],
                  'ccp_alpha':[0.05]
                  }

grid = GridSearchCV(DecisionTreeRegressor(), param_grid=DT_grid_params, cv=4)
grid.fit(X_train, y_train)
DT_grid_best_params = grid.best_params_
DT_grid_best_model = grid.best_estimator_
print('Grid best parameters :', DT_grid_best_params)



dict_of_models_DT = {'DT_base_model_1': DT_base_model_1,
                    'DT_base_model_2': DT_base_model_2,
                    'DT_base_model_3': DT_base_model_3,
                    'DT_random_best_model': DT_random_best_model,
                    'DT_grid_best_model': DT_grid_best_model,
                    }

for name, model in dict_of_models_DT.items():
    search_cv_evaluation(model)
    DT_metrics_matrix.loc[name]= scoring_values

print(DT_metrics_matrix)


"""
There is also no improvement whatsoever. The results are even worse than without optimizing.
I'm probably missing something here.
I couldn't optimize the model DecisionTree. Let's go on with GBR'
"""

# plt.figure(figsize=(200,200))
# plot_tree(DT_base_model_1,
#           filled=True)
# plt.show()

# =================================================================================
#------------------------------------------------------PART 4 : OPTIMISATION OF GBR
# =================================================================================

print('-------------------------------------------------------------PART 4 : OPTIMISATION OF GBR')

GBR_base_model_1 = GradientBoostingRegressor(random_state=6)
GBR_base_model_2 = GradientBoostingRegressor(random_state=7)
GBR_base_model_3 = GradientBoostingRegressor(random_state=8)


GBR_metrics_matrix = pd.DataFrame(index = ['GBR_base_model_1', 'GBR_base_model_2', 'GBR_base_model_3', 'GBR_random_best_model', 'GBR_grid_best_model'],
                                  columns = ['Train_score',
                                             'Test_score',
                                             'r2',
                                             'MAE (absolute values)',
                                             'MAE (%)',
                                             'Accuracy (%)',
                                             'Predictions under 10% error (%)',
                                             'Maximum error (%)'])

#------------------------------------------------------TRAIN TEST SPLIT
print('-------------------------------------------------------------TRAIN TEST SPLIT')

trainset, testset = train_test_split(df, test_size=0.2, random_state=1)

X_train, y_train = variable_selection(trainset)
X_test, y_test = variable_selection(testset)
y_pred = y_test

#------------------------------------------------------EVALUATION FUNCTION
print('-------------------------------------------------------------EVALUATION FUNCTION')

def search_cv_evaluation(model):
    model_fit_predict()
    scoring()
    
    
#------------------------------------------------------BASE MODEL VS RANDOMIZED SEARCH CV
print('-------------------------------------------------------------BASE MODEL VS RANDOMIZED SEARCH CV')

GBR_random_params = {'n_estimators': [10, 50, 100, 200, 500],
                     'max_depth': [2, 3, 5, 10, 15, 20, 50, None],
                     'learning_rate': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                     'loss': ['ls', 'lad', 'huber'],
                     'subsample':[0.2, 0.5, 0.8, 1.0],
                     'n_iter_no_change':[5, 10, 20, None],
                     'max_leaf_nodes':[3,5,10,20,None]
                     }

random = RandomizedSearchCV(GradientBoostingRegressor(), param_distributions=GBR_random_params, cv=4, n_iter=100)
random.fit(X_train, y_train)
GBR_random_best_params = random.best_params_
GBR_random_best_model = random.best_estimator_
print('Random best parameters :', GBR_random_best_params)

# search_cv_evaluation(GBR_base_model_1)
# GBR_metrics_matrix.iloc[0] = scoring_values

# search_cv_evaluation(GBR_base_model_2)
# GBR_metrics_matrix.iloc[1] = scoring_values

# search_cv_evaluation(GBR_base_model_3)
# GBR_metrics_matrix.iloc[2] = scoring_values

# print(GBR_metrics_matrix)

"""

"""

#------------------------------------------------------BASE MODEL VS GRID SEARCH CV
print('-------------------------------------------------------------BASE MODEL VS GRID SEARCH CV')

GBR_grid_params = {'n_estimators': [75, 100, 150],
                     'max_depth': [15],
                     'learning_rate': [0.15, 0.2, 0.25,],
                     'loss': ['ls'],
                     'subsample':[0.8, 1.0],
                     'n_iter_no_change':[None],
                     'max_leaf_nodes':[5, 7, 10]
                     }

grid = GridSearchCV(GradientBoostingRegressor(), param_grid=GBR_grid_params, cv=4)
grid.fit(X_train, y_train)
GBR_grid_best_params = grid.best_params_
GBR_grid_best_model = grid.best_estimator_
print('Grid best parameters :', GBR_grid_best_params)



dict_of_models_GBR = {'GBR_base_model_1': GBR_base_model_1,
                    'GBR_base_model_2': GBR_base_model_2,
                    'GBR_base_model_3': GBR_base_model_3,
                    'GBR_random_best_model': GBR_random_best_model,
                    'GBR_grid_best_model': GBR_grid_best_model,
                    }

for name, model in dict_of_models_GBR.items():
    search_cv_evaluation(model)
    GBR_metrics_matrix.loc[name]= scoring_values

print(GBR_metrics_matrix)


"""
GBR_grid_best_model is the best model we've had :
    - no overfitting : test_score (0.882) almost equal to train_score (0.896)
    - highest accuracy (93.9%) and lowest MAE (6.1%)
    - highest percentage (84.1%) of predictions under 10% of error in MAE

"""


