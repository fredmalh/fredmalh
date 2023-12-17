# -*- coding: utf-8 -*-
"""
10/11/2020
Frédéric Malharin

Analysis of a dataset related to SARS-COV2 with Python in Jupyter Notebooks and Spyder
The goal is to predict the result of the boolean variable "SARS-Cov-2 exam result"

This dataset comes from the course Machine Learnia (https://machinelearnia.com)
Many thanks to Guillaume Saint-Cirgue


FIRST PART : EXPLORATION OF THE DATASET

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

df = pd.read_excel(r'C:\Users\frede\Dropbox\DATA SCIENCE\PORTFOLIO\SARS-COV2-dataset\dataset.xlsx')
backup = df.copy()


#------------------------------------------------------EXPLORATORY ANALYSIS

df.head()
df.shape
df.dtypes.value_counts()
pd.set_option('display.max_row', 111)

# Nan analysis
df.isna()
plt.figure(figsize=(20,15))
sns.heatmap(df.isna(), cbar=False)

missing_rate = df.isna().sum()/df.shape[0]
missing_rate.sort_values()

# Removing Nans
df = df[df.columns[missing_rate < 0.9]]
df = df.drop('Patient ID', axis=1)
df.shape
missing_rate = df.isna().sum()/df.shape[0]

plt.figure(figsize=(20,15))
sns.heatmap(df.isna(), cbar=False)

"""
NOTES :
Target variable = "SARS-Cov-2 exam result"
df.shape = (5644, 111)
df.dtypes.value_counts() = 74 quantitative (float and integers), 37 qualitative
Lots of Nan. I removed all the variables with more than 90% Nan and 'Patient ID'.
df.shape = (5644, 38)
df.dtypes.value_counts() = 18 quantitative (float and integers), 20 qualitative
There are 2 main groups of variables, one with 76% of Nan and one with 89% of Nan.
"""


#------------------------------------------------------VARIABLE ANALYSIS

# Target variable
df["SARS-Cov-2 exam result"].value_counts(normalize=True)

# Patient age quantile
sns.distplot(df['Patient age quantile'])

# Float variables
for column in df.select_dtypes('float'):
    plt.figure()
    sns.distplot(df[column])

df["SARS-Cov-2 exam result"].unique()
for column in df.select_dtypes('object'):
    print(f'{column :-<30} {df[column].unique()}')
   
# Objects variables
for column in df.select_dtypes('object'):
    plt.figure()
    df[column].value_counts().plot.pie()
    
positive_df = df[df['SARS-Cov-2 exam result'] == 'positive']
negative_df = df[df['SARS-Cov-2 exam result'] == 'negative']
blood_columns = df.columns[(missing_rate < 0.9) & (missing_rate > 0.88)]
viral_columns = df.columns[(missing_rate < 0.88) & (missing_rate > 0.75)]

# Blood variables versus target
for column in blood_columns:
    plt.figure()
    sns.distplot(positive_df[column], label = 'positive')
    sns.distplot(negative_df[column], label = 'negative')
    plt.legend()

# Viral variables versus target

for column in viral_columns:
    plt.figure()
    sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'], df[column]), annot=True, fmt='d')  


for column in df:
    df[column] = df[column].replace(['positive','detected'],'1')
    df[column] = df[column].replace(['negative','not_detected'],'0')

df['SARS-Cov-2 exam result'].unique()
for column in df.select_dtypes('object'):
    print(f'{column :-<30} {df[column].unique()}')

# Patient age quantile versus target
sns.countplot(x='Patient age quantile', hue='SARS-Cov-2 exam result', data=df)

"""
NOTES :
Target variable : 10% of positive results to the SARS-COV2 test

Patient age quantile : They are age tranches, could be 1 quantile for 5 yearss. We don't know.

Float variables : they are results of blood analysis. I'll call them blood variables.
All of the float variables are centered on zero. The data have been standardized.

Objects variable : They are booleans, results of several viruses detection test. I'll call them viral variables.
Most results to the tests are widely negative, except one : Rhinovirus/Enterovirus
One variable (Parainfluenza B) has zero case, only not detected. It is useless for our study.

Blood variables versus target : It seems that there is a difference in Platelets, Leukocytes and Monocytes between the groups positive and negative to SARS-COV-2

Viral variables versus target : There's only a few people that are tested positive to another virus on top of SARS-Cov-2'
It seems that there are lots of people infected with Rhinovrius/Enterovirus. Impossible to relaate it to SARS-Cov-2 for now.

Patient age quantile versus target : it seems that all the age categories don't have the same positive rate.
Hard to use given that we don't know much about this Patient age quantile variable.

Overall observation variables versus target : There are only a few variables that seem to be connected to SARS-Cov-2
"""

#------------------------------------------------------DETAILED VARIABLE ANALYSIS

# blood variables versus blood variables
sns.pairplot(df[blood_columns])
sns.heatmap(df[blood_columns].corr())


# blood variables versus Patient age quantile
for column in blood_columns:
    plt.figure()
    sns.lmplot(x='Patient age quantile', y=column, hue='SARS-Cov-2 exam result', data=df)

df.corr()['Patient age quantile'].sort_values()

# Viral variables versus viral variables

pd.crosstab(df['Influenza A'], df['Influenza A, rapid test'])
pd.crosstab(df['Influenza B'], df['Influenza B, rapid test'])

# positive results to any viral test
np.sum(df[viral_columns[:-2]] == '1', axis=1)
np.sum(df[viral_columns[:-2]] == '1', axis=1).plot()
np.sum(df[viral_columns[:-2]] == '1', axis=1) >= 1
df['is_sick'] = np.sum(df[viral_columns[:-2]] == '1', axis=1) >= 1

# Viral variables versus blood variables
df_sick = df[df['is_sick'] == True]
df_not_sick = df[df['is_sick'] == False]

for column in blood_columns:
    plt.figure()
    sns.distplot(df_sick[column], label='sick')
    sns.distplot(df_not_sick[column], label='not sick')
    plt.legend()

def hospital(df):
    if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
        return 'regular ward'
    elif df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
        return 'semi-intensive unit'
    elif df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
        return 'intensive care unit'
    else:
        return 'unknown'

df['status'] = df.apply(hospital, axis=1)

df['status'].unique()

for column in blood_columns:
    plt.figure()
    for category in df['status'].unique():
        sns.distplot(df[df['status']==category][column], label=category)
    plt.legend()


"""
NOTES :
blood variables versus blood variables : some blood variables are correlated (r>0.9)
blood variables versus Patient age quantile : No correlation (<0.28)
Viral variables versus viral variables: Influenza A&B rapid tests are not reliable. We might leave those variables aside.

Viral variables versus blood variables:
    Platelets, moocytes and leukocytes are the same for sick and not sick people, unlike in Blood variables versus target.
    Lymphocytes seem to be different for sick and not sick people
    There are differences in the blood results (lymphocytes) for people in different hopsital categories.
"""

#------------------------------------------------------DETAILED Nan ANALYSIS

df.dropna().count()

df1 = df[viral_columns [:-2]]
df1['SARS-Cov-2 exam result'] = df['SARS-Cov-2 exam result']
df1.dropna()['SARS-Cov-2 exam result'].value_counts(normalize=True)

df2 = df[blood_columns [:-2]]
df2['SARS-Cov-2 exam result'] = df['SARS-Cov-2 exam result']
df2.dropna()['SARS-Cov-2 exam result'].value_counts(normalize=True)

"""
If we drop all the Nan, we're left with 99 patients.
Blood columns 600 patients, viral columns 1300 patients, but we need both.
Viral columns, 92%/8% positive to SARS-COV-2 
Blood columns, 86%/14% positive to SARS-COV-2
"""

#------------------------------------------------------STATISTICAL ANALYSIS OF OUR OBSERVATIONS WITH STUDENT'S TEST

# First hypothesis: the blood variables are the same for patients positive and negative to SARS-Cov-2

from scipy.stats import ttest_ind
positive_df.shape
negative_df.shape
balanced_neg = negative_df.sample(positive_df.shape[0])

def t_test(column):
    alpha = 0.02
    stat, p = ttest_ind(balanced_neg[column].dropna(), positive_df[column].dropna())
    if p < alpha:
        return 'H0 Rejected'
    else:
        return 0

for column in blood_columns:
    print(f'{column :-<50} {t_test(column)}')


"""
The hypothesis is rejected for Platelets, Leukocytes, Eosinophils and Monocytes --> The concentration of those celles in the blood of patients positive to SARS-Cov-2 is different from the one of patients negative to SARS-Cov-2
"""


