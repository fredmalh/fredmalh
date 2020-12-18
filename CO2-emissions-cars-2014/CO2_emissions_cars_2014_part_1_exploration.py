# -*- coding: utf-8 -*-
"""
23/11/2020

Analysis of a dataset related to fuel consumption and CO2 emissions of cars in 2014.
With Python 3.8 (IDE : Jupyter Notebooks)
The goal is to analyse the relationships between the variables (especially the vehicle class)
and the CO2 consumption. I'll train ML models to try to predict CO2 emisions based on the other variables.

PART 1 - EXPLORATION OF THE DATASET AND VARIABLES
"""
  

import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv(r"C:\Users\Fred\Dropbox\DATA SCIENCE\PORTFOLIO\CO2-emissions-cars-2014\FuelConsumptionCo2.csv")
backup = df

# ===============================================================================
#------------------------------------------------------PART 1 : BASIC EXPLORATION
# ===============================================================================

print('-------------------------------------------------------------PART 1 : BASIC EXPLORATION')

print(df.shape)
pd.set_option('display.max_row', 20)
pd.set_option('display.max_column', 13)
print(df)

#------------------------------------------------------NAN ANALYSIS
print('-------------------------------------------------------------NAN ANALYSIS')


df.isna().sum()
df_nan = df[df.isna().any(axis=1)]
print(df_nan)

"""The 3 NaN are on the same row. We drop it."""
df = df.dropna(axis=0)
df.shape

#------------------------------------------------------VARIABLE EXPLORATION
print('-------------------------------------------------------------VARIABLE EXPLORATION')


print(df["MODELYEAR"].unique())
""" "MODELYEAR" has only one value (2014). We can drop it, there's no use for it.¶"""
df = df.drop(["MODELYEAR"], axis=1)

df.dtypes.value_counts()

for column in df.select_dtypes('float64', 'int64').columns:
    plt.figure()
    sns.distplot(df[column])

"""
We've got 4 different fuel consumption : city, highway, combined, and combined MPG (miles per gallon).
I guess that the unit of the first 3 ones is liters per 100km, which is the standard except for the US where it is MPG.
We don't need that many variables, I will keep only the combined in liters per 100km and drop the other 3.¶
"""

df = df.drop(["FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB_MPG"], axis=1)
# print(df)


for column in df.select_dtypes('object').columns:
    print(df[column].value_counts())

"""
There are many categories, especially for 'MODEL' and 'MAKE'.
I'll code these categories into integers to ease the analysis of relationships between variables.
"""

#------------------------------------------------------CODING
print('-------------------------------------------------------------CODING')


code={}

for column in df.select_dtypes('object').columns:
    code[f"code_{column}"] = {}
    list1 = list(df[column].unique())
    i=0
    for val in list1:
        code[f"code_{column}"][f"{val}"] = i
        i+=1

# print(code)

for column in df.select_dtypes('object').columns:
    df.loc[:, column] = df[column].map(code[f"code_{column}"])

# print(df)


#------------------------------------------------------TARGET
print('-------------------------------------------------------------TARGET')

print(sorted(df['CO2EMISSIONS'].unique()))

"""
The target is a discrete variable --> regression algorithms.

Alright, we now have a clean dataset with only integers and floats, and no NaNs.
Let's look at the relationships between variables.
"""


#------------------------------------------------------RELATIONSHIP BETWEEN VARIABLES
print('-------------------------------------------------------------RELATIONSHIP BETWEEN VARIABLES')


# sns.pairplot(df)
sns.heatmap(df.corr())

"""
We can see several strong correlations.
MAKE with MODEL : correlation irrelevant because it was created by me when coding the variables
ENGINESIZE with CYLINDERS correlation irrelevant because it's normal that there are more cylinders when the engine is bigger.
FUELCONSUMPTION_COMB with CO2EMISSIONS are logicallay strongly correlated. I will not use FUELCONSUMPTION_COMB as a variable in the model, I want to be able to predict the CO2 emissions based on the other variables, without knowing the fuel consumption. Though it looks like we have 3 clusters, probably due to the FUELTYPE.
I'll focus on the variables VEHICLECLASS, ENGINESIZE, CYLINDERS, and FUELTYPE.
"""

print(df.corr())
"""
There seem to be no correlation between the fuel type and the CO2 emissions (correlation coefficient of -0.092021).
I'll keep studying the other 3 variables that seem to influence CO2 emissions (coeff of 0.485, 0.874, and 0.847).
"""

df_features = df[['VEHICLECLASS', 'ENGINESIZE', 'CYLINDERS']]
     
for col in df_features:
    plt.figure(figsize=(12,8))
    x = df_features[col]
    y = df["CO2EMISSIONS"]
    plt.scatter(x=x, y=y, c=df["FUELTYPE"])
    b, m = polyfit(x, y, 1)
    plt.plot(x, b+m*x, linewidth=3)
    plt.xlabel(col)
    plt.ylabel("CO2EMISSIONS")
    plt.show()

"""
It looks nice but I don't get more info out of it. I left FUELTYPE as color to see some possible clusters,
but no, there is indeed no relationship between FUELTYPE and CO2EMISSIONS.
"""

plt.figure(figsize=(12,8))
x = df["FUELCONSUMPTION_COMB"]
y = df["CO2EMISSIONS"]
plt.scatter(x=x, y=y, c=df["FUELTYPE"])
# b, m = polyfit(x, y, 1)
# plt.plot(x, b+m*x, linewidth=3)
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()

"""
This was just for fun, I just wanted to check out if FUELTYPE (4 colors) was responsible for the 3 clusters
of data in CO2EMISSIONS vs FUELCONSUMPTION_COMB. It looks like it's the case.
Anyway, now I leave FUELTYPE and FUELCONSUMPTION_COMB aside, we don't need them.

"""

fig = plt.figure(figsize=(10,6))
ax = plt.axes(projection='3d')

xline = np.linspace(-5, 20)
yline = np.linspace(0, 10)
zline = np.linspace(0, 15)

ax.set_xlim3d(-5, 20)
ax.set_ylim3d(0, 10)
ax.set_zlim3d(0, 15)

xdata = df.VEHICLECLASS
ydata = df.ENGINESIZE
zdata = df.CYLINDERS
qdata = df.CO2EMISSIONS

ax.set_xlabel('VEHICLECLASS')
ax.set_ylabel('ENGINESIZE')
ax.set_zlabel('CYLINDERS')
ax.set_title('Relationship between VEHICLECLASS, ENGINESIZE, CYLINDERS and CO2EMISSIONS')

visu3D = ax.scatter3D(xdata, ydata, zdata, c=qdata, s=20, cmap='YlOrBr')
fig.colorbar(visu3D)
plt.show()

"""
It looks good but I don't get more info out of it.
"""
